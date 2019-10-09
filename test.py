from model.custom_lstm import *

if __name__ == "__main__":
	lstm = OptimizedLSTM(100, 32)


	a = torch.arange(5 * 10 * 100).view((5, 10, 100))

	hs, _ = lstm(a.float())

	print(hs.shape)
