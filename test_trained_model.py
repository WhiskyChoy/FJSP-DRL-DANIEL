import time
import os
from common_utils import *
from params import MyConfig, configs
from tqdm import tqdm                           # type: ignore
from data_utils import pack_data_from_config
from model.PPO import PPO_initialize
from common_utils import setup_seed
from fjsp_env_same_op_nums import FJSPEnvForSameOpNums
from typing import List, Optional, Tuple

# question: what if the model is trained with various op nums rather than same op nums?
# answer: because when we do the test, the env list contains only one env, so we can regard it to have same op nums

os.environ["CUDA_VISIBLE_DEVICES"] = configs.device_id
import torch

device = torch.device(configs.device)

ppo = PPO_initialize()
test_time = time.strftime("%Y%m%d_%H%M%S", time.localtime(time.time()))


def test_greedy_strategy(data_set: Tuple[List[np.ndarray], List[np.ndarray]],
                         model_path: str,
                         seed: int,
                         idx: Optional[int],
                         save_path: Optional[str] = None):
    """
        test the model on the given data using the greedy strategy
    :param data_set: test data
    :param model_path: the path of the model file
    :param seed: the seed for testing
    :return: the test results including the makespan and time
    """

    test_result_list = []

    setup_seed(seed)
    ppo.policy.load_state_dict(torch.load(model_path, map_location='cuda'))
    ppo.policy.eval()

    n_j = data_set[0][0].shape[0]
    # n_op, n_m = data_set[1][0].shape
    _, n_m = data_set[1][0].shape
    
    env = FJSPEnvForSameOpNums(n_j=n_j, n_m=n_m)

    suffix = '' if idx is None else f'@({idx})'

    len_data_set = len(data_set[0])
    my_range = range(len_data_set)
    if len_data_set > 1 and configs.test_log_outer_loop:
        my_range = tqdm(my_range, file=sys.stdout, desc=f"progress{suffix}", colour='blue')

    for i in my_range:
        state = env.set_initial_data([data_set[0][i]], [data_set[1][i]])
        t1 = time.time()

        leave = False if len_data_set > 1 and configs.test_log_outer_loop else True  # leave这里是“留下”，不是“离开”
        progress_info = f"({i+1}/{len_data_set})" if len_data_set > 1 else ''
        inner_tqdm = tqdm(file=sys.stdout, desc=f"inner progress{progress_info}{suffix}", colour='blue', total=env.number_of_ops, leave=leave)

        while True:

            with torch.no_grad():
                pi, _ = ppo.policy(fea_j=state.fea_j_tensor,  # [1, N, 8]
                                   op_mask=state.op_mask_tensor,  # [1, N, N]
                                   candidate=state.candidate_tensor,  # [1, J]
                                   fea_m=state.fea_m_tensor,  # [1, M, 6]
                                   mch_mask=state.mch_mask_tensor,  # [1, M, M]
                                   comp_idx=state.comp_idx_tensor,  # [1, M, M, J]
                                   dynamic_pair_mask=state.dynamic_pair_mask_tensor,
                                   fea_pairs=state.fea_pairs_tensor)  # [1, J, M]

            action = greedy_select_action(pi)
            # state, reward, done = env.step(actions=action.cpu().numpy())
            state, _, done = env.step(actions=action.cpu().numpy())
            
            inner_tqdm.update(1)
            inner_tqdm.set_postfix_str(f"makespan: {env.current_makespan[0]}")

            if done:
                inner_tqdm.close()      # remember to close it to release the flush
                break
        t2 = time.time()

        test_result_list.append([env.current_makespan[0], t2 - t1])

        if save_path is not None and configs.test_immediate_save:
            # save the test result
            np.save(save_path, np.array(test_result_list))

    return np.array(test_result_list)


def test_sampling_strategy(data_set: Tuple[List[np.ndarray], List[np.ndarray]],
                           model_path: str,
                           sample_times: int,
                           seed: int,
                           idx: Optional[int] = None,
                           save_path: Optional[str] = None):
    """
        test the model on the given data using the sampling strategy
    :param data_set: test data
    :param model_path: the path of the model file
    :param seed: the seed for testing
    :return: the test results including the makespan and time
    """
    setup_seed(seed)
    test_result_list = []
    ppo.policy.load_state_dict(torch.load(model_path, map_location='cuda'))
    ppo.policy.eval()

    n_j = data_set[0][0].shape[0]
    # n_op, n_m = data_set[1][0].shape
    _, n_m = data_set[1][0].shape
    # from fjsp_env_same_op_nums import FJSPEnvForSameOpNums
    env = FJSPEnvForSameOpNums(n_j, n_m)

    suffix = '' if idx is None else f'@({idx})'

    len_data_set = len(data_set[0])
    my_range = range(len_data_set)
    if len_data_set > 1 and configs.test_log_outer_loop:
        my_range = tqdm(my_range, file=sys.stdout, desc=f"progress{suffix}", colour='blue')

    for i in my_range:
        # copy the testing environment
        JobLength_dataset = np.tile(np.expand_dims(data_set[0][i], axis=0), (sample_times, 1))      # 重复多个相同环境
        OpPT_dataset = np.tile(np.expand_dims(data_set[1][i], axis=0), (sample_times, 1, 1))        # 重复多个相同环境

        state = env.set_initial_data(JobLength_dataset, OpPT_dataset)   # type: ignore
        t1 = time.time()

        leave = False if len_data_set > 1 and configs.test_log_outer_loop else True
        progress_info = f"({i}/{len_data_set})" if len_data_set > 1 else ''
        inner_tqdm = tqdm(file=sys.stdout, desc=f"inner progress{progress_info}{suffix}", colour='blue', total=env.number_of_ops, leave=leave)

        while True:

            with torch.no_grad():
                pi, _ = ppo.policy(fea_j=state.fea_j_tensor,  # [100, N, 8]
                                   op_mask=state.op_mask_tensor,  # [100, N, N]
                                   candidate=state.candidate_tensor,  # [100, J]
                                   fea_m=state.fea_m_tensor,  # [100, M, 6]
                                   mch_mask=state.mch_mask_tensor,  # [100, M, M]
                                   comp_idx=state.comp_idx_tensor,  # [100, M, M, J]
                                   dynamic_pair_mask=state.dynamic_pair_mask_tensor,  # [100, J, M]
                                   fea_pairs=state.fea_pairs_tensor)  # [100, J, M]

            action_envs, _ = sample_action(pi)
            state, _, done = env.step(action_envs.cpu().numpy())

            inner_tqdm.update(1)
            inner_tqdm.set_postfix_str(f"makespan(mean): {env.current_makespan.mean()}")

            if done.all():
                break

        t2 = time.time()
        best_makespan = np.min(env.current_makespan)                # 多个相同环境里最好的
        test_result_list.append([best_makespan, t2 - t1])

        if save_path is not None and configs.test_immediate_save:
            # save the test result
            np.save(save_path, np.array(test_result_list))

    return np.array(test_result_list)


def main(config: MyConfig):
    """
        test the trained model following the config and save the results

        注: 这里传入config其实跟全局的configs是一样的
    """
    flag_sample = config.use_sample

    setup_seed(config.seed_test)
    if not os.path.exists('./test_results'):
        os.makedirs('./test_results')

    # collect the path of test models
    test_model = []

    for model_name in config.test_model:
        test_model.append((f'./trained_network/{config.model_source}/{model_name}.pth', model_name))

    # collect the test data
    # data_source是一级目录，test_data是二级目录
    test_data = pack_data_from_config(config.data_source, config.test_data)

    if flag_sample:
        model_prefix = "DANIELS"
    else:
        model_prefix = "DANIELG"

    test_greedy_n_times = config.test_greedy_n_times

    for data in test_data:          # 过一遍文件夹(可能有多个文件夹)
        print("-" * 25 + "Test Learned Model" + "-" * 25)
        print(f"test data name: {data[1]}")
        print(f"test data length: {len(data[0])}")
        print(f"test mode: {model_prefix}")
        save_direc = f'./test_results/{config.data_source}/{data[1]}'
        if not os.path.exists(save_direc):
            os.makedirs(save_direc)

        for model in test_model:    # 过一遍模型(可能有多个模型)
            save_path = save_direc + f'/Result_{model_prefix}+{model[1]}_{data[1]}.npy'
            # 如果文件路径不存在，或者覆盖标志为True，则进行测试（默认覆盖）
            if (not os.path.exists(save_path)) or config.cover_flag:
                print(f"Model name : {model[1]}")
                print(f"data name: ./data/{config.data_source}/{data[1]}")

                if not flag_sample:
                    print("Test mode: Greedy")
                    result_n_times = []
                    # Greedy mode, test 5 times, record average time.
                    for j in range(test_greedy_n_times):
                        idx = j if test_greedy_n_times > 1 else None
                        tmp_save_path = save_path if config.test_immediate_save else None
                        result = test_greedy_strategy(data[0], model[0], config.seed_test, idx, tmp_save_path)
                        # ↑所有测试都在这个函数完成(对同个data&model)
                        # -- 也就是说，如果该函数内部的循环没有结束，就无法写入实验记录文件
                        result_n_times.append(result)
                    result_n_times_np = np.array(result_n_times)

                    save_result: np.ndarray = np.mean(result_n_times_np, axis=0)
                    print("testing results:")
                    print(f"avg makespan(greedy): ", save_result[:, 0].mean())
                    print(f"avg time: ", save_result[:, 1].mean())

                else:
                    # Sample mode, test once.
                    print("Test mode: Sample")
                    tmp_save_path = save_path if config.test_immediate_save else None
                    save_result = test_sampling_strategy(data[0], model[0], config.sample_times, config.seed_test, None, tmp_save_path)
                    print("testing results:")
                    print(f"avg makespan(sampling): ", save_result[:, 0].mean())
                    print(f"avg time: ", save_result[:, 1].mean())
                np.save(save_path, save_result)


if __name__ == '__main__':
    main(configs)        # type: ignore