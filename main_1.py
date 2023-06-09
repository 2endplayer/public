history = []
    with open('/data.csv', newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='"')
        for row in reader:
            history.append(row)
            

    # -1 is current candle

    database = []

    for candle in history:

        index = 0
        for value in candle:
            if index == 4:
                database.append(float(value))
            index += 1

    return database


def calculate_profitability(candle_data):
    holding = False
    buy_price = 0
    total_percentage = 0

    for candle in candle_data:
        signal = int(candle[1])
        close_price = float(candle[0])

        if signal == 1 and not holding:
            buy_price = close_price
            holding = True
        elif signal == 2 and holding:
            sell_price = close_price
            holding = False
            profit = sell_price - buy_price
            profit_percentage = (profit / buy_price) * 100
            total_percentage += profit_percentage

        # If signal is 0 or holding is True (i.e., already bought but waiting to sell), do nothing

    return total_percentage


def find_extrema(prices: List[float], delta: float, window: int) -> List[List[float]]:
    extrema_points = []
    for i in range(window, len(prices) - window):
        local_max = max(prices[i - window:i + window + 1])
        local_min = min(prices[i - window:i + window + 1])
        if prices[i] >= local_max - delta:
            extrema_points.append([i, prices[i], 2])  # high point
        elif prices[i] <= local_min + delta:
            extrema_points.append([i, prices[i], 1])  # low point
        else:
            extrema_points.append([i, prices[i], 0])  # no action

    return extrema_points


# ---------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------


class MyEnv(gym.Env):
    def __init__(self):
        # initialize your environment here
        self.database = download_market_data()
        self.observation_space = spaces.Box(low=np.array([0.0, 0]), high=np.array([10, 25]), dtype=np.float32)
        self.action_space = spaces.Discrete(2)
        self.current_step = 0
        self.max_steps = 1000

    def step(self, action):
        # take an action (which corresponds to selecting some random parameters)
        # and return the next state, reward, and done flag
        err_msg = f"{action!r} ({type(action)}) invalid"
        assert self.action_space.contains(action), err_msg
        assert self.current_step < self.max_steps, "Max steps reached. Call reset to start again."

        self.current_step += 1

        self.observation_space.sample()

        database = find_extrema(self.database, self.observation_space.sample()[0], self.observation_space.sample()[1])
        reward = calculate_profitability(database)

        done = self.current_step == self.max_steps

        return self.observation_space.sample(), reward, done, {}

    def reset(self, **kwargs):
        # reset the environment to its initial state
        self.current_step = 0
        return self.observation_space.sample()

    def render(self, mode='human'):
        print(f"params: {self.params}, profit: {self.profit}")


# ---------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------

env = MyEnv()

# Define the number of actions and the number of observations
nb_actions = env.action_space.n
nb_observations = env.observation_space.shape[0]

# Define the model architecture
model = tf.keras.models.Sequential()
model.add(InputLayer(input_shape=(nb_observations,)))
model.add(Dense(64, activation="relu"))
model.add(Dense(64, activation="relu"))
model.add(Dense(nb_actions, activation="linear"))

# Define the memory buffer
memory = SequentialMemory(limit=1000000, window_length=1)

# Define the policy for choosing actions
policy = EpsGreedyQPolicy(0.1)

# Define the optimizer
optimizer = Adam(lr=0.001)

# Define the DQN agent
dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=1000,
               target_model_update=1e-2, policy=policy, enable_double_dqn=True, enable_dueling_network=True,
               dueling_type="avg")

# Compile the DQN agent with optimizer
dqn.compile(optimizer=Adam(lr=0.001))

# Train the DQN agent
dqn.fit(env, nb_steps=10000, visualize=False, verbose=2)
