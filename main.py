import pygame
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from torch.utils.tensorboard import SummaryWriter
from collections import deque
import logging
import os

# --------------------------- Constants and Configuration ---------------------------

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')

# Screen dimensions
WIDTH, HEIGHT = 800, 600
WIN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("2D Volleyball Game with Advanced AI")

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (34, 139, 34)  # Forest Green
RED = (178, 34, 34)    # Firebrick Red
BLUE = (65, 105, 225)  # Royal Blue

# Game variables
FPS = 60
GRAVITY = 0.5
BALL_RADIUS = 15
PLAYER_WIDTH = 50  # Reduced width for more precise movements
PLAYER_HEIGHT = 60  # Increased height to resemble real players

NET_WIDTH, NET_HEIGHT = 10, int(HEIGHT * 0.25)

# Fonts
pygame.font.init()
SCORE_FONT = pygame.font.SysFont('Arial', 30)

# Initialize clock
clock = pygame.time.Clock()

# Device configuration for MPS (Apple Silicon), CUDA, or CPU
if torch.backends.mps.is_available():
    device = torch.device("mps")
    logging.info("Using MPS device")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    logging.info("Using CUDA device")
else:
    device = torch.device("cpu")
    logging.info("Using CPU device")

# Neural Network parameters
INPUT_SIZE = 18  # Expanded to include more features
HIDDEN_SIZE = 256  # Increased hidden size
OUTPUT_SIZE = 6   # Actions: Move Left, Stay, Move Right, Jump, Jump+Move Left, Jump+Move Right
LEARNING_RATE = 1e-4  # Adjusted learning rate

# Training parameters
EPISODES = 100000
GAMMA = 0.99
CLIP_EPSILON = 0.2
UPDATE_EPOCHS = 5
BATCH_SIZE = 64
SAVE_INTERVAL = 5000     # Save models every 5000 episodes
RALLY_THRESHOLD = 100    # Rally length threshold for stopping
CONSISTENCY_WINDOW = 50  # Number of consecutive episodes to check for rally threshold
DISPLAY_INTERVAL = 1     # Display game window every DISPLAY_INTERVAL steps

# Define MAX_STEPS_PER_EPISODE
MAX_STEPS_PER_EPISODE = 2000  # Increased steps per episode

# Reward scaling
SCORE_REWARD = 10    # Reward for winning a point
LOSS_PENALTY = -10   # Penalty for losing a point
HIT_REWARD = 1       # Reward for hitting the ball
RALLY_REWARD = 0.1   # Steady reward for maintaining rally
NET_PENALTY = -0.2   # Penalty for staying too close to the net
IDLE_PENALTY = -0.05 # Penalty for idle actions
POSITION_REWARD = 0.1  # Reward for optimal positioning

# Player movement parameters
PLAYER_SPEED = 7     # Increased speed for more dynamic movements
JUMP_VELOCITY = -15

# Initialize TensorBoard writer
writer = SummaryWriter('runs/volleyball_ai')

# Action mapping
ACTION_SPACE = [0, 1, 2, 3, 4, 5]  # 0: Move Left, 1: Stay, 2: Move Right, 3: Jump, 4: Jump+Move Left, 5: Jump+Move Right

# Net position
NET_X = WIDTH // 2 - NET_WIDTH // 2
NET_Y = HEIGHT - NET_HEIGHT

# Model saving paths
MODEL_DIR = 'models'
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

# --------------------------- Neural Network Definition ---------------------------

class ActorCritic(nn.Module):
    def __init__(self):
        super(ActorCritic, self).__init__()
        # Common network
        self.fc_common = nn.Sequential(
            nn.Linear(INPUT_SIZE, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
            nn.ReLU(),
        )
        # Actor network
        self.fc_actor = nn.Sequential(
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, OUTPUT_SIZE),
            nn.Softmax(dim=-1),
        )
        # Critic network
        self.fc_critic = nn.Sequential(
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, 1),
        )

    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32, device=device)
        common = self.fc_common(x)
        action_probs = self.fc_actor(common)
        state_value = self.fc_critic(common)
        return action_probs, state_value

# --------------------------- Player Class Definition ---------------------------

class Player:
    def __init__(self, x, color, side, agent):
        self.x = x
        self.y = HEIGHT - PLAYER_HEIGHT
        self.vel_x = 0
        self.vel_y = 0
        self.color = color
        self.on_ground = True
        self.side = side  # 'left' or 'right'
        self.agent = agent
        self.memory = []      # For PPO
        self.rewards = []     # Initialize rewards
        self.optimizer = None # To be assigned externally
        self.reward = 0       # Current step reward
        self.last_action = 1  # Start with 'Stay'

    def get_state(self, ball_x, ball_y, ball_vel_x, ball_vel_y, opponent_x, opponent_y):
        # Additional features: distance to ball, ball direction, previous action
        distance_x = (ball_x - self.x) / WIDTH
        distance_y = (ball_y - self.y) / HEIGHT
        ball_direction = 1 if ball_vel_x > 0 else -1
        opponent_distance_x = (opponent_x - self.x) / WIDTH
        opponent_distance_y = (opponent_y - self.y) / HEIGHT
        net_distance = (NET_X - self.x) / WIDTH if self.side == 'left' else (self.x - (NET_X + NET_WIDTH)) / WIDTH
        state = np.array([
            (self.x - WIDTH / 2) / (WIDTH / 2),
            (self.y - HEIGHT / 2) / (HEIGHT / 2),
            self.vel_x / PLAYER_SPEED,
            self.vel_y / abs(JUMP_VELOCITY),
            (ball_x - WIDTH / 2) / (WIDTH / 2),
            (ball_y - HEIGHT / 2) / (HEIGHT / 2),
            ball_vel_x / 10,
            ball_vel_y / 10,
            (opponent_x - WIDTH / 2) / (WIDTH / 2),
            (opponent_y - HEIGHT / 2) / (HEIGHT / 2),
            opponent_distance_x,
            opponent_distance_y,
            net_distance,
            1 if self.on_ground else -1,
            distance_x,
            distance_y,
            ball_direction,
            self.last_action / (OUTPUT_SIZE - 1)
        ], dtype=np.float32)
        return state

    def select_action(self, state):
        action_probs, state_value = self.agent(state)
        dist = torch.distributions.Categorical(action_probs)
        action = dist.sample()
        self.memory.append({
            'state': state,
            'action': action,
            'log_prob': dist.log_prob(action),
            'value': state_value,
        })
        self.last_action = action.item()
        return action.item()

    def move(self, action):
        old_x = self.x  # Store the old position
        old_on_ground = self.on_ground

        # Actions: 0 = Move Left, 1 = Stay, 2 = Move Right, 3 = Jump, 4 = Jump+Move Left, 5 = Jump+Move Right
        if action == 0:
            self.vel_x = -PLAYER_SPEED
            self.attempt_jump(False)
        elif action == 1:
            self.vel_x = 0
            self.attempt_jump(False)
        elif action == 2:
            self.vel_x = PLAYER_SPEED
            self.attempt_jump(False)
        elif action == 3:
            self.vel_x = 0
            self.attempt_jump(True)
        elif action == 4:
            self.vel_x = -PLAYER_SPEED
            self.attempt_jump(True)
        elif action == 5:
            self.vel_x = PLAYER_SPEED
            self.attempt_jump(True)
        else:
            self.vel_x = 0
            self.attempt_jump(False)

        # Update position
        self.update()

        # Penalize for ineffective actions
        if self.x == old_x and self.on_ground == old_on_ground:
            self.reward += IDLE_PENALTY  # Small penalty for ineffective actions

    def attempt_jump(self, should_jump):
        if should_jump and self.on_ground:
            self.jump()

    def jump(self):
        if self.on_ground:
            self.vel_y = JUMP_VELOCITY
            self.on_ground = False

    def update(self):
        # Apply gravity
        if not self.on_ground:
            self.vel_y += GRAVITY
            self.y += self.vel_y
            if self.y >= HEIGHT - PLAYER_HEIGHT:
                self.y = HEIGHT - PLAYER_HEIGHT
                self.vel_y = 0
                self.on_ground = True
        else:
            self.vel_y = 0

        # Update horizontal position
        self.x += self.vel_x
        self.keep_within_bounds()

    def keep_within_bounds(self):
        if self.side == 'left':
            self.x = max(0, min(self.x, NET_X - PLAYER_WIDTH))
        else:
            self.x = max(NET_X + NET_WIDTH, min(self.x, WIDTH - PLAYER_WIDTH))

    def draw(self, win):
        # Change color based on current reward
        if self.reward < 0:
            current_color = RED  # Negative reward
        elif self.reward > 0:
            current_color = GREEN  # Positive reward
        else:
            current_color = self.color  # Neutral
        pygame.draw.rect(win, current_color, (int(self.x), int(self.y), PLAYER_WIDTH, PLAYER_HEIGHT))

# --------------------------- Game Mechanics Functions ---------------------------

def draw_window():
    WIN.fill(BLACK)

    # Draw net
    pygame.draw.rect(WIN, WHITE, (NET_X, NET_Y, NET_WIDTH, NET_HEIGHT))

    # Draw players
    left_player.draw(WIN)
    right_player.draw(WIN)

    # Draw ball
    pygame.draw.circle(WIN, BLUE, (int(ball_x), int(ball_y)), BALL_RADIUS)

    # Draw scores
    left_score_text = SCORE_FONT.render(f"Player 1: {left_player_score}", True, WHITE)
    right_score_text = SCORE_FONT.render(f"Player 2: {right_player_score}", True, WHITE)
    WIN.blit(left_score_text, (10, 10))
    WIN.blit(right_score_text, (WIDTH - right_score_text.get_width() - 10, 10))

    pygame.display.flip()

def ball_movement():
    global ball_x, ball_y, ball_vel_x, ball_vel_y
    global left_player_score, right_player_score, done
    global rally_length, last_player_hit

    # Apply gravity
    ball_vel_y += GRAVITY
    ball_x += ball_vel_x
    ball_y += ball_vel_y

    # Floor collision (ball hits the ground)
    if ball_y + BALL_RADIUS >= HEIGHT:
        if ball_x < WIDTH // 2:
            # Ball hit the ground on the left side
            right_player_score += 1
            left_player.reward += LOSS_PENALTY  # Penalty for losing point
            right_player.reward += SCORE_REWARD   # Reward for winning point
            logging.info("Ball hit the ground on the left side. Left player penalized.")
        else:
            left_player_score += 1
            right_player.reward += LOSS_PENALTY  # Penalty for losing point
            left_player.reward += SCORE_REWARD   # Reward for winning point
            logging.info("Ball hit the ground on the right side. Right player penalized.")
        done = True
        return  # Skip rest of function to avoid errors

    # Ceiling collision
    if ball_y - BALL_RADIUS <= 0:
        ball_y = BALL_RADIUS
        ball_vel_y *= -1

    # Wall collision
    if ball_x - BALL_RADIUS <= 0:
        ball_x = BALL_RADIUS
        ball_vel_x *= -1
    if ball_x + BALL_RADIUS >= WIDTH:
        ball_x = WIDTH - BALL_RADIUS
        ball_vel_x *= -1

    # Net collision
    if NET_X <= ball_x <= NET_X + NET_WIDTH:
        if ball_y + BALL_RADIUS >= NET_Y and ball_y - BALL_RADIUS <= NET_Y:
            # Bounce on top
            ball_y = NET_Y - BALL_RADIUS
            ball_vel_y *= -1
        elif ball_y + BALL_RADIUS > NET_Y:
            # Hit side of the net
            if ball_x < WIDTH // 2:
                right_player_score += 1
                left_player.reward += LOSS_PENALTY
                right_player.reward += SCORE_REWARD
                logging.info("Ball hit the net on the left side. Left player penalized.")
            else:
                left_player_score += 1
                right_player.reward += LOSS_PENALTY
                left_player.reward += SCORE_REWARD
                logging.info("Ball hit the net on the right side. Right player penalized.")
            done = True
            return

        # Prevent the ball from getting stuck
        if ball_vel_x > 0:
            ball_x = NET_X + NET_WIDTH + BALL_RADIUS
            ball_vel_x = abs(ball_vel_x) * 0.8  # Slight damping
        else:
            ball_x = NET_X - BALL_RADIUS
            ball_vel_x = -abs(ball_vel_x) * 0.8  # Slight damping

    # Player collision
    collision_occurred = False
    for player in [left_player, right_player]:
        if (player.x <= ball_x <= player.x + PLAYER_WIDTH and
            player.y <= ball_y + BALL_RADIUS <= player.y + PLAYER_HEIGHT):
            # Adjust ball position and velocity
            ball_y = player.y - BALL_RADIUS
            ball_vel_y = -abs(ball_vel_y)  # Ensure the ball goes upward
            # Adjust x velocity based on hit position and player's velocity
            hit_pos = (ball_x - (player.x + PLAYER_WIDTH / 2)) / (PLAYER_WIDTH / 2)
            ball_vel_x = hit_pos * 8 + player.vel_x * 0.5  # Adjust x velocity

            if last_player_hit == player.side:
                # Penalize for consecutive hits
                player.reward += LOSS_PENALTY * 0.5  # Smaller penalty
                logging.info(f"{player.side.capitalize()} player penalized for consecutive hit.")
            else:
                # Positive reward for hitting the ball
                player.reward += HIT_REWARD

            last_player_hit = player.side
            collision_occurred = True
            rally_length += 1
            logging.info(f"Rally {rally_length}: Ball hit by {player.side} player.")
            break  # Only one player can hit the ball at a time

    if collision_occurred:
        # Introduce a steady rally reward to encourage longer rallies
        rally_reward = RALLY_REWARD
        left_player.reward += rally_reward
        right_player.reward += rally_reward

    # Introduce positional rewards and penalties
    for player in [left_player, right_player]:
        # Penalize for being too close to the net
        if player.side == 'left' and player.x >= NET_X - PLAYER_WIDTH - 10:
            player.reward += NET_PENALTY
        elif player.side == 'right' and player.x <= NET_X + NET_WIDTH + 10:
            player.reward += NET_PENALTY

        # Reward for being in optimal position relative to the ball
        optimal_position = (ball_x - player.x) / WIDTH
        position_reward = -abs(optimal_position) * POSITION_REWARD
        player.reward += position_reward

def reset_ball():
    global ball_x, ball_y, ball_vel_x, ball_vel_y, done, rally_length, last_player_hit
    
    # Randomly choose a side (left or right)
    side = random.choice(['left', 'right'])
    
    if side == 'left':
        # Drop on the left side
        ball_x = random.uniform(WIDTH * 0.1, NET_X - BALL_RADIUS)
    else:
        # Drop on the right side
        ball_x = random.uniform(NET_X + NET_WIDTH + BALL_RADIUS, WIDTH * 0.9)
    
    ball_y = HEIGHT * 0.2  # Start higher up
    ball_vel_x = random.uniform(-5, 5)
    ball_vel_y = random.uniform(-5, 0)  # Initial downward velocity, but more random
    done = False
    rally_length = 0
    last_player_hit = None

def compute_gae(rewards, values, gamma=GAMMA, lam=0.95):
    advantages = []
    gae = 0
    for i in reversed(range(len(rewards))):
        delta = rewards[i] + gamma * values[i + 1] - values[i]
        gae = delta + gamma * lam * gae
        advantages.insert(0, gae)
    return advantages

def ppo_update(player):
    if len(player.memory) < 2:
        return  # Not enough data to update

    # Separate step-wise memory and the last value
    step_memory = player.memory[:-1]
    last_value = player.memory[-1]['value'].detach().squeeze()

    # Gather data from step_memory
    states = torch.tensor([item['state'] for item in step_memory], dtype=torch.float32, device=device)
    actions = torch.tensor([item['action'] for item in step_memory], dtype=torch.long, device=device)
    old_log_probs = torch.stack([item['log_prob'] for item in step_memory]).detach()
    values = torch.stack([item['value'] for item in step_memory]).detach().squeeze()
    rewards = torch.tensor(player.rewards, dtype=torch.float32, device=device)

    # Compute advantages using GAE
    values = torch.cat((values, last_value.unsqueeze(0)), dim=0)  # Append last value
    advantages = compute_gae(rewards, values)

    if len(advantages) == 0:
        logging.warning("No advantages computed. Skipping PPO update.")
        return

    # Convert advantages and returns to tensors
    advantages = torch.tensor(advantages, dtype=torch.float32, device=device)
    returns = advantages + values[:-1]

    # Normalize advantages
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    # Perform PPO update
    dataset = torch.utils.data.TensorDataset(states, actions, old_log_probs, returns, advantages)
    loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    for _ in range(UPDATE_EPOCHS):
        for batch in loader:
            b_states, b_actions, b_old_log_probs, b_returns, b_advantages = batch

            action_probs, state_values = player.agent(b_states)
            dist = torch.distributions.Categorical(action_probs)
            entropy = dist.entropy().mean()

            new_log_probs = dist.log_prob(b_actions)
            ratio = torch.exp(new_log_probs - b_old_log_probs)

            surr1 = ratio * b_advantages
            surr2 = torch.clamp(ratio, 1 - CLIP_EPSILON, 1 + CLIP_EPSILON) * b_advantages

            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = nn.MSELoss()(state_values.squeeze(), b_returns)
            loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy  # Entropy to encourage exploration

            player.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(player.agent.parameters(), 0.5)
            player.optimizer.step()

    # Clear memory and rewards after update
    player.memory = []
    player.rewards = []

# --------------------------- Main Training Function ---------------------------

def main():
    global ball_x, ball_y, ball_vel_x, ball_vel_y
    global left_player_score, right_player_score, done
    global rally_length, last_player_hit

    # Assign optimizers to players
    left_player.optimizer = left_optimizer
    right_player.optimizer = right_optimizer

    episode = 0
    rally_threshold_reached = False

    while episode < EPISODES and not rally_threshold_reached:
        episode += 1
        reset_ball()
        left_player.x = 100
        right_player.x = WIDTH - 100 - PLAYER_WIDTH
        left_player.y = HEIGHT - PLAYER_HEIGHT
        right_player.y = HEIGHT - PLAYER_HEIGHT
        left_player.vel_x = left_player.vel_y = 0
        right_player.vel_x = right_player.vel_y = 0
        left_player.on_ground = True
        right_player.on_ground = True

        left_player.memory = []
        right_player.memory = []
        left_player.rewards = []
        right_player.rewards = []

        step = 0
        total_reward_left = 0
        total_reward_right = 0

        left_player_score = 0
        right_player_score = 0
        rally_length = 0

        done = False

        while not done and step < MAX_STEPS_PER_EPISODE:
            clock.tick(FPS)
            step += 1

            # Handle events to prevent freezing
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

            # Reset rewards for this step
            left_player.reward = 0
            right_player.reward = 0

            # Get states
            state_left = left_player.get_state(ball_x, ball_y, ball_vel_x, ball_vel_y, right_player.x, right_player.y)
            state_right = right_player.get_state(ball_x, ball_y, ball_vel_x, ball_vel_y, left_player.x, left_player.y)

            # Select actions
            action_left = left_player.select_action(state_left)
            action_right = right_player.select_action(state_right)

            # Move players
            left_player.move(action_left)
            right_player.move(action_right)

            # Update players
            left_player.update()
            right_player.update()

            # Move ball
            ball_movement()

            # Accumulate rewards
            total_reward_left += left_player.reward
            total_reward_right += right_player.reward

            # Append rewards to player.rewards
            left_player.rewards.append(left_player.reward)
            right_player.rewards.append(right_player.reward)

            # Check if rally threshold is reached
            if rally_length > RALLY_THRESHOLD:
                rally_threshold_reached = True
                logging.info(f"Rally threshold reached at Episode {episode} with Rally Length: {rally_length}")
                break

            # Draw window every DISPLAY_INTERVAL steps
            if step % DISPLAY_INTERVAL == 0:
                draw_window()

        # Append last value for GAE
        with torch.no_grad():
            _, last_value_left = left_player.agent(left_player.get_state(ball_x, ball_y, ball_vel_x, ball_vel_y, right_player.x, right_player.y))
            _, last_value_right = right_player.agent(right_player.get_state(ball_x, ball_y, ball_vel_x, ball_vel_y, left_player.x, left_player.y))

        left_player.memory.append({'value': last_value_left})
        right_player.memory.append({'value': last_value_right})

        # Update agents
        ppo_update(left_player)
        ppo_update(right_player)

        # Logging
        avg_reward_left = total_reward_left / step if step > 0 else 0
        avg_reward_right = total_reward_right / step if step > 0 else 0

        logging.info(f"Episode {episode} - Left Reward: {avg_reward_left:.2f}, Right Reward: {avg_reward_right:.2f}, "
                     f"Rally Length: {rally_length}")

        # Log to TensorBoard
        writer.add_scalar('Reward/Left', avg_reward_left, episode)
        writer.add_scalar('Reward/Right', avg_reward_right, episode)
        writer.add_scalar('Rally Length', rally_length, episode)
        writer.add_scalar('Episode Length', step, episode)

        episode_rewards.append((avg_reward_left, avg_reward_right))
        episode_lengths.append(step)
        episode_rally_lengths.append(rally_length)
        recent_rallies.append(rally_length)

        # Check for consistent rally over recent episodes
        if len(recent_rallies) == recent_rallies.maxlen:
            if all(r >= RALLY_THRESHOLD for r in recent_rallies):
                rally_threshold_reached = True
                logging.info(f"Consistently achieved rally length >{RALLY_THRESHOLD} over the last {recent_rallies.maxlen} episodes.")
                break

        # Save models at intervals
        if episode % SAVE_INTERVAL == 0:
            try:
                torch.save(left_agent.state_dict(), os.path.join(MODEL_DIR, f"left_agent_episode_{episode}.pth"))
                torch.save(right_agent.state_dict(), os.path.join(MODEL_DIR, f"right_agent_episode_{episode}.pth"))
                logging.info(f"Episode {episode}: Models saved.")
            except Exception as e:
                logging.error(f"Error saving models at Episode {episode}: {e}")

    if rally_threshold_reached:
        logging.info(f"Training stopped at Episode {episode} after achieving a rally length of {rally_length}.")
    else:
        logging.info(f"Reached maximum episodes ({EPISODES}) without achieving the rally threshold.")

    # After training, watch the agents play
    watch_agents_play()

# --------------------------- Watch Mode Function ---------------------------

def watch_agents_play():
    global ball_x, ball_y, ball_vel_x, ball_vel_y, done, rally_length, last_player_hit
    done = False
    rally_length = 0
    reset_ball()

    left_player.x = 100
    right_player.x = WIDTH - 100 - PLAYER_WIDTH
    left_player.y = HEIGHT - PLAYER_HEIGHT
    right_player.y = HEIGHT - PLAYER_HEIGHT
    left_player.vel_x = left_player.vel_y = 0
    right_player.vel_x = right_player.vel_y = 0
    left_player.on_ground = True
    right_player.on_ground = True

    left_player.memory = []
    right_player.memory = []
    left_player.rewards = []
    right_player.rewards = []

    step = 0

    # Load models if available
    left_model_path = os.path.join(MODEL_DIR, 'left_agent_final.pth')
    right_model_path = os.path.join(MODEL_DIR, 'right_agent_final.pth')
    if os.path.exists(left_model_path):
        left_agent.load_state_dict(torch.load(left_model_path))
        logging.info("Loaded left agent model.")
    if os.path.exists(right_model_path):
        right_agent.load_state_dict(torch.load(right_model_path))
        logging.info("Loaded right agent model.")

    while not done and step < MAX_STEPS_PER_EPISODE:
        clock.tick(FPS)
        step += 1

        # Handle events to prevent freezing
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        # Get states
        state_left = left_player.get_state(ball_x, ball_y, ball_vel_x, ball_vel_y, right_player.x, right_player.y)
        state_right = right_player.get_state(ball_x, ball_y, ball_vel_x, ball_vel_y, left_player.x, left_player.y)

        # Select actions
        with torch.no_grad():
            action_probs_left, _ = left_player.agent(state_left)
            dist_left = torch.distributions.Categorical(action_probs_left)
            action_left = dist_left.sample().item()

            action_probs_right, _ = right_player.agent(state_right)
            dist_right = torch.distributions.Categorical(action_probs_right)
            action_right = dist_right.sample().item()

        # Move players
        left_player.move(action_left)
        right_player.move(action_right)

        # Update players
        left_player.update()
        right_player.update()

        # Move ball
        ball_movement()

        # Draw window
        draw_window()

    logging.info("Watch mode ended.")

# --------------------------- Initialize Players ---------------------------

# Initialize agents
left_agent = ActorCritic().to(device)
right_agent = ActorCritic().to(device)

# Load models if available
left_model_path = os.path.join(MODEL_DIR, 'left_agent_final.pth')
right_model_path = os.path.join(MODEL_DIR, 'right_agent_final.pth')
if os.path.exists(left_model_path):
    left_agent.load_state_dict(torch.load(left_model_path))
    logging.info("Loaded left agent model.")
if os.path.exists(right_model_path):
    right_agent.load_state_dict(torch.load(right_model_path))
    logging.info("Loaded right agent model.")

# Initialize players
left_player = Player(100, GREEN, 'left', left_agent)
right_player = Player(WIDTH - 100 - PLAYER_WIDTH, GREEN, 'right', right_agent)

# Assign optimizers to players
left_optimizer = optim.Adam(left_agent.parameters(), lr=LEARNING_RATE)
right_optimizer = optim.Adam(right_agent.parameters(), lr=LEARNING_RATE)
left_player.optimizer = left_optimizer
right_player.optimizer = right_optimizer

# Initialize scores and rally length
left_player_score = 0
right_player_score = 0
rally_length = 0
done = False
last_player_hit = None

# Initialize episode tracking
episode_rewards = []
episode_lengths = []
episode_rally_lengths = []
recent_rallies = deque(maxlen=CONSISTENCY_WINDOW)  # For monitoring consistency

# Initialize ball variables
ball_x = WIDTH // 2
ball_y = HEIGHT // 2
ball_vel_x = random.uniform(-5, 5)
ball_vel_y = -12

# --------------------------- Run the Script ---------------------------

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
        pygame.quit()
        sys.exit()
