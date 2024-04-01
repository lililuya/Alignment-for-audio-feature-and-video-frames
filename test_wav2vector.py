from transformers import Wav2Vec2Processor, Wav2Vec2Model
import torch
import librosa
import datetime


# 加载预训练的模型和处理器
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")


"""
wav2vector和mel的关系
"""
# 加载音频文件
audio_file_path = "/mnt/sdb/cxh/liwen/Imitator/assets/demo/audio1.wav"  
speech_array, sampling_rate = librosa.load(audio_file_path, sr=16000)

duration = librosa.get_duration(speech_array, sampling_rate)
duration_in_ms = duration * 1000
time = str(datetime.timedelta(milliseconds=duration_in_ms))
print(time)  # 0:00:03.833375

# print(speech_array)
# 使用处理器加载音频文件并提取特征, 这个padding可能会带来哪些影响
# When used in normal mode, this method forwards all its arguments to Wav2Vec2FeatureExtractor’s __call__()
"""
Wav2Vec2FeatureExtractor的call方法
- raw_speech (np.ndarray, List[float], List[np.ndarray], List[List[float]]) – The sequence or batch of sequences to be padded. Each sequence can be a numpy array, a list of float values, a list of numpy arrays or a list of list of float values.
- padding (bool, str or PaddingStrategy, optional, defaults to False) –
  Select a strategy to pad the returned sequences (according to the model’s padding side and padding index) among:
    - True or 'longest': Pad to the longest sequence in the batch (or no padding if only a single sequence if provided).
    - 'max_length': Pad to a maximum length specified with the argument max_length or to the maximum acceptable input length for the model if that argument is not provided.
    - False or 'do_not_pad' (default): No padding (i.e., can output a batch with sequences of different lengths).
- max_length (int, optional) – Maximum length of the returned list and optionally padding length (see above).
- pad_to_multiple_of (int, optional) –
  If set will pad the sequence to a multiple of the provided value.
  This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >= 7.5 (Volta), or on TPUs which benefit from having sequence lengths be a multiple of 128.
- return_attention_mask (bool, optional) –
  Whether to return the attention mask. If left to the default, will return the attention mask according to the specific feature_extractor’s default.
- return_tensors (str or TensorType, optional) –
  If set, will return tensors instead of list of python integers. Acceptable values are:
   - 'tf': Return TensorFlow tf.constant objects.
   - 'pt': Return PyTorch torch.Tensor objects.
   - 'np': Return Numpy np.ndarray objects.
- sampling_rate (int, optional) – The sampling rate at which the raw_speech input was sampled. It is strongly recommended to pass sampling_rate at the forward call to prevent silent errors.
- padding_value (float, defaults to 0.0)
"""
input_values = processor(speech_array, return_tensors="pt", padding=True) # batch=1的话，padding无效

# 获取输入特征
input_values = input_values.input_values
print(model)
"""
Wav2Vec2Model(
  (feature_extractor): Wav2Vec2FeatureEncoder(
    (conv_layers): ModuleList(
      (0): Wav2Vec2GroupNormConvLayer(
        (conv): Conv1d(1, 512, kernel_size=(10,), stride=(5,), bias=False)
        (activation): GELUActivation()
        (layer_norm): GroupNorm(512, 512, eps=1e-05, affine=True)
      )
      (1-4): 4 x Wav2Vec2NoLayerNormConvLayer(
        (conv): Conv1d(512, 512, kernel_size=(3,), stride=(2,), bias=False)
        (activation): GELUActivation()
      )
      (5-6): 2 x Wav2Vec2NoLayerNormConvLayer(
        (conv): Conv1d(512, 512, kernel_size=(2,), stride=(2,), bias=False)
        (activation): GELUActivation()
      )
    )
  )
  # 5*2*2*2*2*2*2 = 320
  # 320/16000=0.02
  (feature_projection): Wav2Vec2FeatureProjection(
    (layer_norm): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
    (projection): Linear(in_features=512, out_features=768, bias=True)
    (dropout): Dropout(p=0.1, inplace=False)
  )
  (encoder): Wav2Vec2Encoder(
    (pos_conv_embed): Wav2Vec2PositionalConvEmbedding(
      (conv): ParametrizedConv1d(
        768, 768, kernel_size=(128,), stride=(1,), padding=(64,), groups=16
        (parametrizations): ModuleDict(
          (weight): ParametrizationList(
            (0): _WeightNorm()
          )
        )
      )
      (padding): Wav2Vec2SamePadLayer()
      (activation): GELUActivation()
    )
    (layer_norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
    (dropout): Dropout(p=0.1, inplace=False)
    (layers): ModuleList(
      (0-11): 12 x Wav2Vec2EncoderLayer(
        (attention): Wav2Vec2Attention(
          (k_proj): Linear(in_features=768, out_features=768, bias=True)
          (v_proj): Linear(in_features=768, out_features=768, bias=True)
          (q_proj): Linear(in_features=768, out_features=768, bias=True)
          (out_proj): Linear(in_features=768, out_features=768, bias=True)
        )
        (dropout): Dropout(p=0.1, inplace=False)
        (layer_norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        (feed_forward): Wav2Vec2FeedForward(
          (intermediate_dropout): Dropout(p=0.1, inplace=False)
          (intermediate_dense): Linear(in_features=768, out_features=3072, bias=True)
          (intermediate_act_fn): GELUActivation()
          (output_dense): Linear(in_features=3072, out_features=768, bias=True)
          (output_dropout): Dropout(p=0.1, inplace=False)
        )
        (final_layer_norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
      )
    )
  )
)
"""
with torch.no_grad():
    features = model(input_values).last_hidden_state

print(input_values.shape)
print(features.shape)


# 提取mel频谱
import audio
import numpy as np

wav = audio.load_wav("/mnt/sdb/cxh/liwen/Imitator/assets/demo/audio1.wav", 16000)
mel = audio.melspectrogram(wav)
print(mel.shape)