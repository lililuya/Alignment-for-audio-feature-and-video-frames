# 🤔Alignment-for-audio-feature-and-video-frames 👍
![Language](https://img.shields.io/badge/language-c++-brightgreen) 
## 😙1.Motivation❓

+ 在Talking Head任务中，需要满足音频视频同步
+ 音频主要送入网络的是音频特征，常见的音频特征

  + 未经过神经网络处理的特征
    + Mel 频谱
    + MFCC系数
  + 经过神经网络处理的特征
    + HuBert特征
    + Wav2Vec2得到的特征
    + Deepspeech特征

## 2.对于Wav2Vec2特征做一个简单的讨论

### 2.1对于测试音频

+ 首先获取音频的具体时长

```python
audio_file_path = "/mnt/sdb/cxh/liwen/Imitator/assets/demo/audio1.wav"  
speech_array, sampling_rate = librosa.load(audio_file_path, sr=16000)

duration = librosa.get_duration(speech_array, sampling_rate)
duration_in_ms = duration * 1000
time = str(datetime.timedelta(milliseconds=duration_in_ms))
print(time)  # 0:00:03.833375
```

### 2.2然后观察Wav2Vec2的网络结构

```python
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
```

### 2.3可以大致分为几个模块：

+ 首先对音频特征进行提取，用了一个`feature_extractor`模块对音频特征进行提取吗，这个地方的话，用到了7个卷积层，对时间序列的缩小倍数是5*2^6=320
+ 后续的处理模块都是对这个时间步得到的特征进行进一步的后处理操作，主要是增加上下文信息、改善特征的表达能力等
+ 官方文档中对于`Wav2Vec2Model`模型返回值的描述

> A [`BaseModelOutput`](https://huggingface.co/transformers/v4.7.0/main_classes/output.html#transformers.modeling_outputs.BaseModelOutput) or a tuple of `torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various elements depending on the configuration ([`Wav2Vec2Config`](https://huggingface.co/transformers/v4.7.0/model_doc/wav2vec2.html#transformers.Wav2Vec2Config)) and inputs.
>
> - **last_hidden_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`) – Sequence of hidden-states at the output of the last layer of the model.
>
> - **hidden_states** (`tuple(torch.FloatTensor)`, optional, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) – Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
>
>   Hidden-states of the model at the output of each layer plus the initial embedding outputs.
>
> - **attentions** (`tuple(torch.FloatTensor)`, optional, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) – Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length, sequence_length)`.

+ 可以看到输出`last_hidden_state `的第二个维度是`sequence_length`
+ 验证猜想，直接在Transformer模块的`Wav2Vec2Model`的``feature_extractor``上输出维度，观察与最后得到的**last_hidden_state** 的维度是否相等

### 2.4对维度进行测试

+ 在Transformer源码模块加上维度信息打印

```python
        hidden_states, extract_features = self.feature_projection(extract_features)
        print("hidden_states",hidden_states.shape)
        print("extract_features", extract_features.shape)
        hidden_states = self._mask_hidden_states(
```

+ 得到维度结果，证明后面模块的处理对`sequence_length`并没有影响

```python
hidden_states 		torch.Size([1, 191, 768])
extract_features 	torch.Size([1, 191, 512])
last_hidden_state 	torch.Size([1, 191, 768])
```



### 2.5序列长度维度与时间的对应关系

+ 根据前面提到的卷积层，特征卷积对音频时间维度缩小倍数为320

+ 也就是说对于`16KHZ`采样率的音频，卷积过后缩小320倍，得到的采样率是`16000/320=50`

+ 也就是如果输入`1s`的音频得到的这个序列长度维度对应的时间应该是`1s/50=0.02s`

+ 最后的`last_hidden_state`的维度也应该是`0.02s`即视频帧的`0.5帧（`25fps）

+ **住意：**

  + 有一个操作会对音频长度进行改变

  + ```python
    input_values = processor(speech_array, return_tensors="pt", padding=True) # batch=1的话，padding无效
    ```

  + 官方文档解释

  + >**When used Wav2Vec2Processor  in normal mode, this method forwards all its arguments to Wav2Vec2FeatureExtractor’s __call__()**
    >
    >- **raw_speech** (`np.ndarray`, `List[float]`, `List[np.ndarray]`, `List[List[float]]`) – The sequence or batch of sequences to be padded. Each sequence can be a numpy array, a list of float values, a list of numpy arrays or a list of list of float values.
    >
    >- **padding** (`bool`, `str` or [`PaddingStrategy`](https://huggingface.co/transformers/v4.7.0/internal/file_utils.html#transformers.file_utils.PaddingStrategy), optional, defaults to `False`) –
    >
    >  Select a strategy to pad the returned sequences (according to the model’s padding side and padding index) among:
    >
    >  - `True` or `'longest'`: Pad to the longest sequence in the batch (or no padding if only a single sequence if provided).
    >  - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum acceptable input length for the model if that argument is not provided.
    >  - `False` or `'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of different lengths).
    >
    >  
    >
    >- **max_length** (`int`, optional) – Maximum length of the returned list and optionally padding length (see above).
    >
    >- **pad_to_multiple_of** (`int`, optional) –
    >
    >  If set will pad the sequence to a multiple of the provided value.
    >
    >  This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >= 7.5 (Volta), or on TPUs which benefit from having sequence lengths be a multiple of 128.
    >
    >  
    >
    >- **return_attention_mask** (`bool`, optional) –
    >
    >  Whether to return the attention mask. If left to the default, will return the attention mask according to the specific feature_extractor’s default.
    >
    >- **return_tensors** (`str` or [`TensorType`](https://huggingface.co/transformers/v4.7.0/internal/file_utils.html#transformers.file_utils.TensorType), optional) –
    >
    >  If set, will return tensors instead of list of python integers. Acceptable values are:
    >
    >  - `'tf'`: Return TensorFlow `tf.constant` objects.
    >  - `'pt'`: Return PyTorch `torch.Tensor` objects.
    >  - `'np'`: Return Numpy `np.ndarray` objects.
    >
    >  
    >
    >- **sampling_rate** (`int`, optional) – The sampling rate at which the `raw_speech` input was sampled. It is strongly recommended to pass `sampling_rate` at the forward call to prevent silent errors.
    >
    >- **padding_value** (`float`, defaults to 0.0) –



### 2.6实例测试

+ 针对上面的方法验证实例
  + 2.1获取的音频长度
    + `0:00:03.833375`
  + 得到的`embedding` `(last_hidden_state)`维度
    + `torch.Size([1, 191, 768])`
  + 计算时间
    + `191*0.02=3.82`
    + 时间能对上



## 3.Mel频谱对齐的探究

+ pass

## 4.MFCC探究

+ pass

## 5.DeepSpeech探究

+ pass
📝License
