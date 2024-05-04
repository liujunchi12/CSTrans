import torch
import torch.nn as nn
import numpy as np

class LMMD_loss(nn.Module):
    def __init__(self, class_num=31, kernel_type='rbf', kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        super(LMMD_loss, self).__init__()
        self.class_num = class_num
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = fix_sigma
        self.kernel_type = kernel_type

    def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0-total1)**2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i)
                          for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp)
                      for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def get_loss(self, source, target, s_label, t_label):
        batch_size = source.size()[0]
        weight_ss, weight_tt, weight_st = self.cal_weight(
            s_label, t_label, batch_size=batch_size, class_num=self.class_num)
        weight_ss = torch.from_numpy(weight_ss).cuda()
        weight_tt = torch.from_numpy(weight_tt).cuda()
        weight_st = torch.from_numpy(weight_st).cuda()

        kernels = self.guassian_kernel(source, target,
                                kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
        loss = torch.Tensor([0]).cuda()
        if torch.sum(torch.isnan(sum(kernels))):
            return loss
        SS = kernels[:batch_size, :batch_size]
        TT = kernels[batch_size:, batch_size:]
        ST = kernels[:batch_size, batch_size:]

        loss += torch.sum(weight_ss * SS + weight_tt * TT - 2 * weight_st * ST)
        return loss

    def convert_to_onehot(self, sca_label, class_num=31):
        return np.eye(class_num)[sca_label]

    def cal_adjust(self, s_label, t_label):
        # print("s_label:", s_label.shape)     #s_label: torch.Size([16])
        # print("s_label:", s_label)           #s_label: tensor([24, 10,  8, 26,  4, 17, 23, 11,  6,  4,  6,  8,  6, 19, 11, 20],device='cuda:0')

        # print("t_label:", t_label.shape)     #t_label: torch.Size([16, 31])
        # print("t_label:", t_label)
        batch_size = s_label.size()[0]
        # print("batch_size:", batch_size)     #batch_size: 16

        s_sca_label = s_label.cpu().data.numpy()
        # print("s_sca_label:", s_sca_label.shape)     #s_sca_label: (16,)
        # print("s_sca_label:", s_sca_label)           #s_sca_label: [24 10  8 26  4 17 23 11  6  4  6  8  6 19 11 20]


        t_sca_label = t_label.cpu().data.max(1)[1].numpy()
        # print("t_sca_label:", t_sca_label.shape)        #t_sca_label: (16,)
        # print("t_sca_label:", t_sca_label)              #t_sca_label: [13  3 28 18 11 27  1 12 24 10 10  3 14  2  7 12]


        index = list(set(s_sca_label) & set(t_sca_label))
        # print("index:", index)                             #index: [24, 10, 11]

        ########################################add ####################################

        # aaa = set(s_sca_label)
        # bbb = set(t_sca_label)
        # print("aaa:", aaa)
        # print("bbb:", bbb)

        co_sca_label = np.append(s_sca_label, t_sca_label)
        # print("co_sca_label:", co_sca_label)

        # s_index = list(enumerate(s_sca_label))
        # t_index = list(enumerate(t_sca_label))
        # co_index = list(enumerate(co_sca_label))

        # print("s_index:", s_index)
        # print("t_index:", t_index)
        # print("co_index:", co_index)

        s_index_adjust = []
        t_index_adjust = []

        for index_in in index:
            count_s = 0
            count_t = 0

            for i, x in enumerate(co_sca_label):
                if x == index_in:
                    if i <= 7:
                        count_s = count_s + 1
                    else:
                        count_t = count_t + 1

                    # print("s_index_in_index_s:", i)
                    if i <= 7:
                        s_index_adjust.append(i)
                    else:
                        t_index_adjust.append(i - 8)
                    # print("s_index_in_value_s:", index_in)
                    # print("s_index_adjust:", s_index_adjust)
                    # print("t_index_adjust:", t_index_adjust)
            # print("count_s:", count_s)
            # print("count_t:", count_t)
            if count_s > count_t:
                count_num = count_s - count_t
                for i in range(count_num):
                    s_index_adjust.pop()
            else:
                count_num = count_t - count_s
                for i in range(count_num):
                    dump = s_index_adjust[-1]
                    s_index_adjust.append(dump)
            # print("s_index_adjust:", s_index_adjust)
            # print("t_index_adjust:", t_index_adjust)
            # count_s = 0
            # count_t = 0
        ########################################add ####################################
        # print("s_index_adjust_final:", s_index_adjust)
        # print("t_index_adjust_final:", t_index_adjust)
        return s_index_adjust, t_index_adjust


    def cal_weight(self, s_label, t_label, batch_size=32, class_num=31):
        # print("s_label:", s_label.shape)     #s_label: torch.Size([16])
        # print("s_label:", s_label)           #s_label: tensor([24, 10,  8, 26,  4, 17, 23, 11,  6,  4,  6,  8,  6, 19, 11, 20],device='cuda:0')


        # print("t_label:", t_label.shape)     #t_label: torch.Size([16, 31])
        # print("t_label:", t_label)
        batch_size = s_label.size()[0]
        # print("batch_size:", batch_size)     #batch_size: 16


        s_sca_label = s_label.cpu().data.numpy()
        # print("s_sca_label:", s_sca_label.shape)     #s_sca_label: (16,)
        # print("s_sca_label:", s_sca_label)           #s_sca_label: [24 10  8 26  4 17 23 11  6  4  6  8  6 19 11 20]


        s_vec_label = self.convert_to_onehot(s_sca_label, class_num=self.class_num)
        # print("s_vec_label:", s_vec_label.shape)      #s_vec_label: (16, 31)
        # print("s_vec_label:", s_vec_label)

        s_sum = np.sum(s_vec_label, axis=0).reshape(1, class_num)
        # print("s_sum:", s_sum.shape)                   #s_sum: (1, 31)
        # print("s_sum:", s_sum)

        s_sum[s_sum == 0] = 100
        # print("s_sum:", s_sum.shape)                   #s_sum: (1, 31)
        # print("s_sum:", s_sum)

        s_vec_label = s_vec_label / s_sum
        # print("s_vec_label:", s_vec_label.shape)       #s_vec_label: (16, 31)
        # print("s_vec_label:", s_vec_label)



        t_sca_label = t_label.cpu().data.max(1)[1].numpy()
        # print("t_sca_label:", t_sca_label.shape)        #t_sca_label: (16,)
        # print("t_sca_label:", t_sca_label)              #t_sca_label: [13  3 28 18 11 27  1 12 24 10 10  3 14  2  7 12]



        t_vec_label = t_label.cpu().data.numpy()
        # print("t_vec_label:", t_vec_label.shape)        #t_vec_label: (16, 31)
        # print("t_vec_label:", t_vec_label)

        t_sum = np.sum(t_vec_label, axis=0).reshape(1, class_num)
        # print("t_sum:", t_sum.shape)                     #t_sum: (1, 31)
        # print("t_sum:", t_sum)

        t_sum[t_sum == 0] = 100
        # print("t_sum:", t_sum.shape)                     #t_sum: (1, 31)
        # print("t_sum:", t_sum)

        t_vec_label = t_vec_label / t_sum
        # print("t_vec_label:", t_vec_label.shape)         #t_vec_label: (16, 31)
        # print("t_vec_label:", t_vec_label)


        index = list(set(s_sca_label) & set(t_sca_label))
        # print("index:", index)                             #index: [24, 10, 11]



        mask_arr = np.zeros((batch_size, class_num))
        # print("mask_arr:", mask_arr.shape)                #mask_arr: (16, 31)
        # print("mask_arr:", mask_arr)

        mask_arr[:, index] = 1
        # print("mask_arr:", mask_arr.shape)                #mask_arr: (16, 31)
        # print("mask_arr:", mask_arr)

        t_vec_label = t_vec_label * mask_arr
        # print("t_vec_label:", t_vec_label.shape)         #t_vec_label: (16, 31)
        # print("t_vec_label:", t_vec_label)

        s_vec_label = s_vec_label * mask_arr
        # print("s_vec_label:", s_vec_label.shape)         #s_vec_label: (16, 31)
        # print("s_vec_label:", s_vec_label)


        weight_ss = np.matmul(s_vec_label, s_vec_label.T)
        # print("weight_ss:", weight_ss.shape)              #weight_ss: (16, 16)
        # print("weight_ss:", weight_ss)


        weight_tt = np.matmul(t_vec_label, t_vec_label.T)
        # print("weight_tt:", weight_tt.shape)              #weight_tt: (16, 16)
        # print("weight_tt:", weight_tt)

        weight_st = np.matmul(s_vec_label, t_vec_label.T)
        # print("weight_st:", weight_st.shape)              #weight_st: (16, 16)
        # print("weight_st:", weight_st)

        length = len(index)
        # print("length:", length)                          #length: 3

        if length != 0:
            weight_ss = weight_ss / length
            # print("weight_ss1:", weight_ss.shape)           #weight_tt1: (16, 16)
            # print("weight_ss1:", weight_ss)

            weight_tt = weight_tt / length
            # print("weight_tt1:", weight_tt.shape)          #weight_tt1: (16, 16)
            # print("weight_tt1:", weight_tt)

            weight_st = weight_st / length
            # print("weight_st1:", weight_st.shape)          #weight_st1: (16, 16)
            # print("weight_st1:", weight_st)

        else:
            weight_ss = np.array([0])
            # print("weight_ss2:", weight_ss.shape)  # none
            # print("weight_ss2:", weight_ss)        # none
            weight_tt = np.array([0])
            weight_st = np.array([0])
        return weight_ss.astype('float32'), weight_tt.astype('float32'), weight_st.astype('float32')