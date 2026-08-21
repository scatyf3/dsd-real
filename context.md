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


但网络怎么做不知道，从zmq到http