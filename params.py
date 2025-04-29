import argparse
from typing import List
import json

class MyConfig(argparse.Namespace):
    # Device configuration
    use_json_config: bool
    device: str
    device_id: str

    # File name configuration
    model_suffix: str
    data_suffix: str

    # AutoExperiment flags
    cover_flag: bool
    cover_data_flag: bool
    cover_heu_flag: bool
    cover_train_flag: bool

    # Data load configuration
    model_source: str
    data_source: str

    # SD2 data generation
    op_per_job: float
    op_per_mch_min: int
    op_per_mch_max: int
    data_size: int
    data_type: str

    # TestData to Excel
    sort_flag: bool

    # OR-Tools configuration
    max_solve_time: int

    # Seed configuration
    seed_datagen: int
    seed_train_vali_datagen: int
    seed_train: int
    seed_test: int

    # Environment configuration
    n_j: int
    n_m: int
    n_op: int
    low: int
    high: int

    # Network configuration
    fea_j_input_dim: int
    fea_m_input_dim: int
    dropout_prob: float
    num_heads_OAB: List[int]
    num_heads_MAB: List[int]
    layer_fea_output_dim: List[int]
    num_mlp_layers_actor: int
    hidden_dim_actor: int
    num_mlp_layers_critic: int
    hidden_dim_critic: int

    # PPO Algorithm configuration
    num_envs: int
    max_updates: int
    lr: float
    gamma: float
    k_epochs: int
    eps_clip: float
    vloss_coef: float
    ploss_coef: float
    entloss_coef: float
    tau: float
    gae_lambda: float

    # Training configuration
    train_size: str
    validate_timestep: int
    reset_env_timestep: int
    minibatch_size: int

    # Test configuration
    test_data: List[str]
    use_sample: bool
    sample_times: int
    test_model: List[str]
    test_method: List[str]

def str2bool(v: str):
    """
        transform string value to bool value
    :param v: a string input
    :return: the bool value
    """
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')

with open('./config.json', 'r', encoding='utf-8') as f:
    json_configs: dict = json.load(f)

parser = argparse.ArgumentParser(description='Arguments for DANIEL_FJSP')
# args for device
parser.add_argument('--use_json_config', type=str2bool, default=False,
                    help='Whether using the default config file')
parser.add_argument('--device', type=str, default='cuda', help='Device name')
parser.add_argument('--device_id', type=str, default='0', help='Device id')

# args for file_name

parser.add_argument('--model_suffix', type=str, default='', help='Suffix of the model')
parser.add_argument('--data_suffix', type=str, default='mix', help='Suffix of the data')

# args for AutoExperiment
parser.add_argument('--cover_flag', type=str2bool, default=True, help='Whether covering test results of the model')
parser.add_argument('--cover_data_flag', type=str2bool, default=False, help='Whether covering the generated data')
parser.add_argument('--cover_heu_flag', type=str2bool, default=False,
                    help='Whether covering test results of heuristics')
parser.add_argument('--cover_train_flag', type=str2bool, default=True, help='Whether covering the trained model')

# args for data load
parser.add_argument('--model_source', type=str, default='SD2', help='Suffix of the data that model trained on')
parser.add_argument('--data_source', type=str, default='SD2', help='Suffix of test data')

# args for SD2 data generation
parser.add_argument('--op_per_job', type=float, default=0,
                    help='Number of operations per job, default 0, means the number equals m')      # int, not float
parser.add_argument('--op_per_mch_min', type=int, default=1,
                    help='Minimum number of compatible machines for each operation')
parser.add_argument('--op_per_mch_max', type=int, default=5,
                    help='Maximum number of compatible machines for each operation')
parser.add_argument('--data_size', type=int, default=100, help='The number of instances for data generation')
parser.add_argument('--data_type', type=str, default="test", help='Generated data type (test/vali)')

# args for testData to excel
parser.add_argument('--sort_flag', type=str2bool, default=True,
                    help='Whether sorting the printed results by the makespan')

# args for or-tools
parser.add_argument('--max_solve_time', type=int, default=1800, help='The maximum solving time of OR-Tools')

# args for seed
parser.add_argument('--seed_datagen', type=int, default=200, help='Seed for data generation')
parser.add_argument('--seed_train_vali_datagen', type=int, default=100, help='Seed for generate validation data')
parser.add_argument('--seed_train', type=int, default=300, help='Seed for training')
parser.add_argument('--seed_test', type=int, default=50, help='Seed for testing heuristics')
# args for tricks

# args for env
parser.add_argument('--n_j', type=int, default=10, help='Number of jobs of the instance')
parser.add_argument('--n_m', type=int, default=5, help='Number of machines of the instance')
parser.add_argument('--n_op', type=int, default=50, help='Number of operations of the instance')
parser.add_argument('--low', type=int, default=1, help='Lower Bound of processing time(PT)')
parser.add_argument('--high', type=int, default=99, help='Upper Bound of processing time')

# args for network
parser.add_argument('--fea_j_input_dim', type=int, default=10, help='Dimension of operation raw feature vectors')
parser.add_argument('--fea_m_input_dim', type=int, default=8, help='Dimension of machine raw feature vectors')

parser.add_argument('--dropout_prob', type=float, default=0.0, help='Dropout rate (1 - keep probability).')

parser.add_argument('--num_heads_OAB', nargs='+', type=int, default=[4, 4],
                    help='Number of attention head of operation message attention block')
parser.add_argument('--num_heads_MAB', nargs='+', type=int, default=[4, 4],
                    help='Number of attention head of machine message attention block')
parser.add_argument('--layer_fea_output_dim', nargs='+', type=int, default=[32, 8],
                    help='Output dimension of the DAN layers')

parser.add_argument('--num_mlp_layers_actor', type=int, default=3, help='Number of layers in Actor network')
parser.add_argument('--hidden_dim_actor', type=int, default=64, help='Hidden dimension of Actor network')
parser.add_argument('--num_mlp_layers_critic', type=int, default=3, help='Number of layers in Critic network')
parser.add_argument('--hidden_dim_critic', type=int, default=64, help='Hidden dimension of Critic network')

# args for PPO Algorithm
parser.add_argument('--num_envs', type=int, default=20, help='Batch size for training environments')
parser.add_argument('--max_updates', type=int, default=1000, help='No. of episodes of each env for training')
parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')

parser.add_argument('--gamma', type=float, default=1, help='Discount factor used in training')
parser.add_argument('--k_epochs', type=int, default=4, help='Update frequency of each episode')
parser.add_argument('--eps_clip', type=float, default=0.2, help='Clip parameter')
parser.add_argument('--vloss_coef', type=float, default=0.5, help='Critic loss coefficient')
parser.add_argument('--ploss_coef', type=float, default=1, help='Policy loss coefficient')
parser.add_argument('--entloss_coef', type=float, default=0.01, help='Entropy loss coefficient')
parser.add_argument('--tau', type=float, default=0, help='Policy soft update coefficient')
parser.add_argument('--gae_lambda', type=float, default=0.98, help='GAE parameter')

# args for training
parser.add_argument('--train_size', type=str, default="10x5", help='Size of training instances')
parser.add_argument('--validate_timestep', type=int, default=10, help='Interval for validation and data log')
parser.add_argument('--reset_env_timestep', type=int, default=20, help='Interval for reseting the environment')
parser.add_argument('--minibatch_size', type=int, default=1024, help='Batch size for computing the gradient')

# args for test
parser.add_argument('--test_data', nargs='+', default=['10x5+mix'], help='List of data for testing')
parser.add_argument('--use_sample', type=str2bool, default=False, help='Whether using the sampling strategy in testing')
parser.add_argument('--sample_times', type=int, default=100, help='Sampling times for the sampling strategy')
parser.add_argument('--test_model', nargs='+', default=['10x5+mix'], help='List of model for testing')
parser.add_argument('--test_method', nargs='+', default=[], help='List of heuristic methods for testing')

configs: MyConfig = parser.parse_args()     # type: ignore

if configs.use_json_config:
    for key, value in json_configs.items():
        if hasattr(configs, key):
            setattr(configs, key, value)
        else:
            print(f"Warning: {key} is not a valid argument. It will be ignored.")