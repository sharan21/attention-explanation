from Transparency.common_code.common import *
from Transparency.common_code.plotting import *
from Transparency.Trainers.DatasetBC import *
import numpy as np
import sys
import matplotlib.pyplot as plt

base_dir1 = 'experiments/new_sst_diversity_lstm_dot/sst/diversity_lstm+dot__diversity_weight_0_0'
base_dir2 = 'experiments/new_sst_diversity_lstm_dot/sst/diversity_lstm+dot__diversity_weight_0.5_0'

dataset = 'sst'

dataset = datasets[dataset]()

def process_grads(grads) :
	for k in grads :
		if k != "conicity":
			xxe = grads[k]
			print ("xxe len shape", len(xxe),xxe[0].shape)
			for i in range(len(xxe)) :
				xxe[i] = np.abs(xxe[i]).sum(0)

def get_outputs(base_dir):
	dirname = get_latest_model(base_dir)
	outputs = pload1(dirname,'test_output')
#     outputs['cd_attn'] = pload1(dirname,'cd')
#     outputs['cd_matrix'] = pload1(dirname,'cd_matrix')
	prob = np.array(outputs['yt_hat']).squeeze()
	yt_pred= np.zeros_like(prob)
	yt_pred[prob>0.5] = 1
	outputs['yt_pred'] = yt_pred
	return outputs


def print_attention(output1, output2):
	threshold = 0
	for idx in range(len(output1['attn_hat'])):

		attn1 = output1['attn_hat'][idx]
		#         attn1_cd = output1['cd_attn'][idx]
		#         cd_matrix1 = np.array(output1['cd_matrix'][idx])

		attn2 = output2['attn_hat'][idx]
		#         attn2_cd = output2['cd_attn'][idx]
		#         cd_matrix2 = np.array(output2['cd_matrix'][idx])

		L = len(output1['X'][idx])
		js_divergence = jsd(attn1[1:L - 1], attn2[1:L - 1])
		if (js_divergence >= threshold):

			sentence = dataset.vec.map2words(output1['X'][idx])
			y = output1['y'][idx]
			L = len(sentence)
			y_pred1 = output1['yt_pred'][idx]
			y_pred2 = output2['yt_pred'][idx]

			if (y == y_pred1 and y == y_pred2):
				print_attn(sentence[1:L - 1], attn1[1:L - 1])
				#                 print_attn(sentence[1:L-1], attn1_cd[1:L-1])

				print_attn(sentence[1:L - 1], attn2[1:L - 1])
				#                 print_attn(sentence[1:L-1], attn2_cd[1:L-1])
				"""
				fig, ax = init_gridspec(2, 2, 2)
				ax[0] = plot_matrix(ax[0], cd_matrix1[1:L-1,1:L-1], sentence[1:L-1], sentence[1:L-1])
				ax[0].invert_yaxis()                

				ax[1] = plot_matrix(ax[1], cd_matrix2[1:L-1,1:L-1], sentence[1:L-1], sentence[1:L-1])
				ax[1].invert_yaxis()                
				show_gridspec()
				"""
				print("-" * 30)
				sys.stdout.flush()


if __name__ == "__main__":
	output1 = get_outputs(base_dir1)
	output2 = get_outputs(base_dir2)

	print_attention(output1, output2)