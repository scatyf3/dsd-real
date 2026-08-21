主要是看看vllm支持不支持了

`RFC #42109「Disaggregated Speculative Decoding with Standalone Parallel Draft Model」`，没有相应实现


---

agent告诉我有vllm插件这个东西，这里的plan是

1. 基于vllm，通过插件增加下面的功能
	1. draft：生成gamma个token，kv回滚(prefix cache上即可)
	2. target：verify gamma token，返回结果，kv回滚，这部分用插件覆盖ngram path
2. target 被多个draft共享
	1. 用一个后端服务包target vllm server，请求路由和 `(draft_id, seq_id)` 的应答分发.
    agent补充：请求归target所有


但网络怎么做不知道，从zmq到http，先包装成抽象层，具体咋干再说

### stage 0 

1. 测试vllm插件

这里想验证的是能否修改vllm里的draft，用custom_class(vllm 0.22.0) 注册一个remotedrafter，现在可以replay一些预先录制的draft sequence。

8×L40S (sm_89, 48GB) 

### stage 1

2. 实现draft和target的传输层，一个抽象的传输层，可以换http（慢）或者zmq/grpc（快）