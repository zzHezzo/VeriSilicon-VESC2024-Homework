import torch
from model import CNN
from pathlib import Path

class VAD(object):
    def __init__(self,model_path=None,sample_rate=8000,frame_len=0.03) -> None:
        """
        :param model_path: model file dir
        :param frame_len: Length of time per frame (s)
        """
        if sample_rate != 8000:
            raise Exception("sampel rate must be 8000 Hz!")

        if (model_path == None) or (not Path(model_path).exists()):
            raise Exception("model path not exists!")

        self.model = None
        self.sample_rate = sample_rate
        self.frame_len = self.frame_len = int(frame_len * self.sample_rate)

        self.__load_model__(model_path=model_path)

    def __load_model__(self,model_path):
        """load model

        :param model_path: model file dir
        :return: The model after loading
        """
        self.model = CNN()
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

    def process(self,inp_data):
        """The actual processing of the Voice Activity Detector, It is processed frame by frame and the results are smoothed

        :param inp_data: Input audio data,(sample_N,)
        :return Current output audio data(N,), the label that can be output currently(M,),The position corresponding to the starting point of each frame(M,)
        """

        if len(inp_data.shape) > 1:
            raise Exception("data shape should be (sample_N,)!")

        if len(inp_data) < self.frame_len:
            raise Exception("input length must be %d".format(self.frame_len))

        frame_data = torch.from_numpy(inp_data).float()
        frame_data = frame_data.unsqueeze(0).unsqueeze(0).unsqueeze(0)

        model_output = self.model(frame_data)
        pred = torch.max(model_output,1)[1].data.numpy()

        return pred
