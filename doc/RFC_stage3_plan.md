# 实验：Liger-Kernel中各kernel相比torch的speed/memory提升比是否大于1

## 测试环境

基础环境
- platform: https://www.autodl.com/
- NPU: Atlas 900 A2 PoD(64G)
- HDK=25.2.0
- CANN=8.5.0
- CPU: 24 vCPU Kunpeng-920
- OS: ubuntu22.04

依赖软件版本匹配
- python=3.10.8
- Liger-Kernel=0.7.0, Commit ID: 781083b2cdead14bdc035058ab52e5d396bb0113
- torch=2.6.0
- torch_npu=2.6.0
- torchvision==0.21.0
- triton-ascend=3.2.0
- transformers=5.2.0


## 实测结果

| kernel name                     | speed forward | speed backward | speed full | memory  |
|:--------------------------------|:-------------:|:--------------:|:----------:|:-------:|
| rms_norm                        |       ❌       |       ❌        |     ❌      |    ✅    |
| swiglu                          |       ✅       |       ❌        |     ❌      |    ✅    |
| rope                            |       ✅       |       ✅        |     ✅      |    ✅    |
| cross_entropy                   |       ❌       |       ✅        |     ❌      |    ✅    |
| softmax                         |       ❌       |       ❌        |     ❌      |    ❌    |
| layer_norm                      |       ❌       |       ❌        |     ❌      |    ✅    |
| group_norm                      |       ❌       |       ❌        |     ❌      |    ✅    |
| geglu                           |       ✅       |       ✅        |     ✅      |    ✅    |
| fused_linear_grpo_loss_sequence |       ❌       |       ✅        |     ❌      |    ❌    |
| fused_linear_grpo_loss_token    |       ❌       |       ✅        |     ❌      |    ❌    |
| dyt_beta=False                  |       ❌       |       ❌        |     ❌      |    ✅    |
| dyt_beta=True                   |       ❌       |       ❌        |     ❌      |    ✅    |
| jsd                             |       ✅       |       ✅        |     ✅      |    ✅    |
| kl_div                          |       ✅       |       ❌        |     ❌      |    ✅    |
| llama4_rope                     |       ✅       |       ✅        |     ✅      |    ✅    |
| poly_norm                       |       ✅       |       ✅        |     ✅      |    ✅    |
| qwen2vl_mrope                   |       ✅       |       ❌        |     ✅      |    ✅    |
| tvd                             |       ✅       |       ✅        |     ✅      |    ✅    |


### 1. High priority

#### 1.1 rms_norm

forward: partial pass ❌
<img width="1000" height="600" alt="rms_norm_speed_forward" src="https://github.com/user-attachments/assets/bb50e59a-5c2b-4595-a2fd-797e1903b2c8" />

backward: partial pass ❌
<img width="1000" height="600" alt="rms_norm_speed_backward" src="https://github.com/user-attachments/assets/291e775a-45a0-4563-aa9a-4b7201a0d8f2" />

full: partial pass ❌
<img width="1000" height="600" alt="rms_norm_speed_full" src="https://github.com/user-attachments/assets/c3954975-45ac-47ad-8528-78abb46237f2" />

memory: ✔
<img width="1000" height="600" alt="rms_norm_memory_full" src="https://github.com/user-attachments/assets/0dfcb4b2-c813-4f37-8086-9aaf4c8b93cd" />


#### 1.2 swiglu

forward: ✔
<img width="1000" height="600" alt="swiglu_speed_forward" src="https://github.com/user-attachments/assets/08dc0c40-132c-47b8-b661-515fc746c730" />

backward: ❌
<img width="1000" height="600" alt="swiglu_speed_backward" src="https://github.com/user-attachments/assets/2b3ca1dc-8e5e-4af0-a087-151a6309ff9d" />

full: ❌
<img width="1000" height="600" alt="swiglu_speed_full" src="https://github.com/user-attachments/assets/fc054b87-f27f-4093-8791-9e948c034012" />

memory: ✔
<img width="1000" height="600" alt="swiglu_memory_full" src="https://github.com/user-attachments/assets/ccd3f20b-4751-49cc-b8ef-f98d7eb6e778" />


#### 1.3 rope

Note: The transformers version is 4.57.3. Related issue: https://github.com/linkedin/Liger-Kernel/issues/1155

forward: ✔
<img width="1000" height="600" alt="rope_speed_forward" src="https://github.com/user-attachments/assets/a9a5131d-3f89-46fe-ade4-176782f29d43" />

backward: ✔
<img width="1000" height="600" alt="rope_speed_backward" src="https://github.com/user-attachments/assets/6a5651c1-790b-433b-b0a1-39d7d365d479" />

full: ✔
<img width="1000" height="600" alt="rope_speed_full" src="https://github.com/user-attachments/assets/385f859e-80b1-41f4-b398-e85ca720f4c2" />

memory: ✔
<img width="1000" height="600" alt="rope_memory_full" src="https://github.com/user-attachments/assets/d2967186-63a0-47fc-8617-d55bf1902922" />


#### 1.4 cross_entropy

forward: ❌
<img width="1000" height="600" alt="cross_entropy_speed_forward" src="https://github.com/user-attachments/assets/6dfc2a1e-0f00-4cbe-8514-e3147d7f829f" />

backward: ✔
<img width="1000" height="600" alt="cross_entropy_speed_backward" src="https://github.com/user-attachments/assets/b7a111f6-d219-4a0c-a3e5-4bcc67689f5c" />

full: ❌
<img width="1000" height="600" alt="cross_entropy_speed_full" src="https://github.com/user-attachments/assets/23f8d9cf-df37-4c69-82e9-f58b1e7832e1" />

no_grad_forward: ❌
<img width="1000" height="600" alt="cross_entropy_speed_no-grad-forward" src="https://github.com/user-attachments/assets/9b94cb21-f4d1-4726-9349-2acf708b175c" />

memory: ✔
<img width="1000" height="600" alt="cross_entropy_memory_full" src="https://github.com/user-attachments/assets/d3880443-d7d5-485e-b005-0982335bf004" />


### 2. Others

#### 2.1 softmax

forward: ❌
<img width="1000" height="600" alt="softmax_speed_forward" src="https://github.com/user-attachments/assets/dd239a46-9206-44f3-b205-c5c40086b367" />

backward: ❌
<img width="1000" height="600" alt="softmax_speed_backward" src="https://github.com/user-attachments/assets/d874b5a3-af60-4757-8033-12e4427cd992" />

full: ❌
<img width="1000" height="600" alt="softmax_speed_full" src="https://github.com/user-attachments/assets/73bfff10-90b9-4c71-91c2-ec3ad62ebda9" />

memory: ❌
<img width="1000" height="600" alt="softmax_memory_full" src="https://github.com/user-attachments/assets/471fcb1d-42f1-482b-aaaa-0a4116164d42" />


#### 2.2 layer_norm

forward: ❌
<img width="1000" height="600" alt="layer_norm_speed_forward" src="https://github.com/user-attachments/assets/352caf82-0897-44b0-a2d1-a0e19fe687b8" />

backward: ❌
<img width="1000" height="600" alt="layer_norm_speed_backward" src="https://github.com/user-attachments/assets/08543474-5f73-4902-8c7b-c4604f3de188" />

full: ❌
<img width="1000" height="600" alt="layer_norm_speed_full" src="https://github.com/user-attachments/assets/36be71f3-c228-4859-82da-458e1f0c58aa" />

memory:✔
<img width="1000" height="600" alt="layer_norm_memory_full" src="https://github.com/user-attachments/assets/cbc19046-654d-49f2-9c0c-1c9c1fac320e" />


#### 2.3 group_norm

forward: ❌
<img width="1000" height="600" alt="group_norm_speed_forward" src="https://github.com/user-attachments/assets/dcd7bd26-7f19-4546-9f86-dc97fa666339" />

backward: ❌
<img width="1000" height="600" alt="group_norm_speed_backward" src="https://github.com/user-attachments/assets/6872ccf9-c4c6-440e-808a-f67410c68019" />

full: ❌
<img width="1000" height="600" alt="group_norm_speed_full" src="https://github.com/user-attachments/assets/8bb29e23-0ee0-4925-847b-751d2e36c81d" />

memory: ✔
<img width="1000" height="600" alt="group_norm_memory_full" src="https://github.com/user-attachments/assets/6e354d8f-2e46-46eb-b97c-2a25d1664a77" />


#### 2.4 geglu

forward: ✔
<img width="1000" height="600" alt="geglu_speed_forward" src="https://github.com/user-attachments/assets/3a2369b6-84a3-483c-8811-27b440cffde9" />

backward: ✔
<img width="1000" height="600" alt="geglu_speed_backward" src="https://github.com/user-attachments/assets/762054ed-8a6c-4fab-aa0f-ac8584085047" />

full: ✔
<img width="1000" height="600" alt="geglu_speed_full" src="https://github.com/user-attachments/assets/cebbdba4-bb7a-414b-80f7-3beca9eec903" />

memory: ✔
<img width="1000" height="600" alt="geglu_memory_full" src="https://github.com/user-attachments/assets/9969ce18-deed-4af2-a001-955c754f3c50" />


#### 2.5 fused_linear_grpo_loss_sequence

forward: ❌
<img width="1000" height="600" alt="fused_linear_grpo_loss_sequence_speed_forward" src="https://github.com/user-attachments/assets/dce0a952-4689-4a3a-ad66-1c4541520590" />

backward: ✔
<img width="1000" height="600" alt="fused_linear_grpo_loss_sequence_speed_backward" src="https://github.com/user-attachments/assets/1acdf3e7-e306-407f-b3b3-47933fbd20eb" />

full: ❌
<img width="1000" height="600" alt="fused_linear_grpo_loss_sequence_speed_full" src="https://github.com/user-attachments/assets/aa44850d-9d49-4b2f-9771-207948c88616" />

memory: ❌
<img width="1000" height="600" alt="fused_linear_grpo_loss_sequence_memory_full" src="https://github.com/user-attachments/assets/dcdb4b2b-e0a8-4257-9037-105fa0121684" />


#### 2.6 fused_linear_grpo_loss_token

forward: ❌
<img width="1000" height="600" alt="fused_linear_grpo_loss_token_speed_forward" src="https://github.com/user-attachments/assets/0321f85e-7c17-4216-a5eb-2f9c0ec223be" />

backward: ✔
<img width="1000" height="600" alt="fused_linear_grpo_loss_token_speed_backward" src="https://github.com/user-attachments/assets/cc52ed39-5574-4b93-af43-95f2816f680d" />

full: ❌
<img width="1000" height="600" alt="fused_linear_grpo_loss_token_speed_full" src="https://github.com/user-attachments/assets/6520ce62-2368-4e15-8d3c-10f598a81bdb" />

memory: ❌
<img width="1000" height="600" alt="fused_linear_grpo_loss_token_memory_full" src="https://github.com/user-attachments/assets/f28ec782-e046-48be-91da-e9a3adffd9be" />


#### 2.7 dyt_beta=False

forward: ❌
<img width="1000" height="600" alt="dyt_beta=False_speed_forward" src="https://github.com/user-attachments/assets/770cda3b-55e1-4e15-90a1-3fd544e50da3" />

backward: ❌
<img width="1000" height="600" alt="dyt_beta=False_speed_backward" src="https://github.com/user-attachments/assets/cf2b6985-2dd5-4987-b2e6-e2614064367c" />

full: ❌
<img width="1000" height="600" alt="dyt_beta=False_speed_full" src="https://github.com/user-attachments/assets/4cd343b6-9222-4ed0-9fe4-348822c3406e" />

memory: ✔
<img width="1000" height="600" alt="dyt_beta=False_memory_full" src="https://github.com/user-attachments/assets/8fe03e83-096c-4dd6-bdef-8b7d8e6e582f" />


#### 2.8 dyt_beta=True

forward: ❌
<img width="1000" height="600" alt="dyt_beta=True_speed_forward" src="https://github.com/user-attachments/assets/486f1a6c-2260-4d1f-8adc-cedb01650208" />

backward: ❌
<img width="1000" height="600" alt="dyt_beta=True_speed_backward" src="https://github.com/user-attachments/assets/b70fad08-6a01-42f9-a6a9-748e39ee43ac" />

full: ❌
<img width="1000" height="600" alt="dyt_beta=True_speed_full" src="https://github.com/user-attachments/assets/e9c549ac-e441-4b19-b766-3877b51a3973" />

memory: ✔
<img width="1000" height="600" alt="dyt_beta=True_memory_full" src="https://github.com/user-attachments/assets/d81170b1-87a1-43df-96fe-b15ff06a13df" />


#### 2.9 jsd

forward: ✔
<img width="1000" height="600" alt="jsd_speed_forward" src="https://github.com/user-attachments/assets/478c09ba-6b90-44a5-abea-a0435292992d" />

backward: ✔
<img width="1000" height="600" alt="jsd_speed_backward" src="https://github.com/user-attachments/assets/cafb999b-5817-41cd-987a-42309d59c431" />

full: ✔
<img width="1000" height="600" alt="jsd_speed_full" src="https://github.com/user-attachments/assets/37a17e4f-9ebf-41b0-8079-d20ba0cf2888" />

memory: ✔
<img width="1000" height="600" alt="jsd_memory_full" src="https://github.com/user-attachments/assets/d4d04fc0-d698-4a13-9dd5-7f74a43f00e4" />


#### 2.10 kl_div

forward: ✔
<img width="1000" height="600" alt="kl_div_speed_forward" src="https://github.com/user-attachments/assets/68e41a8c-9a46-400f-b715-a70aa8a9868d" />

backward: ❌
<img width="1000" height="600" alt="kl_div_speed_backward" src="https://github.com/user-attachments/assets/4ba28e90-91e6-401e-b084-7c746c6bddfa" />

full: ❌
<img width="1000" height="600" alt="kl_div_speed_full" src="https://github.com/user-attachments/assets/870ab4a4-b9c3-4e35-815e-91c417a8eacf" />

memory: ✔
<img width="1000" height="600" alt="kl_div_memory_full" src="https://github.com/user-attachments/assets/069323a0-bc23-44cb-a3a0-ad9aa6a83ab5" />


#### 2.11 llama4_rope

forward: ✔
<img width="1000" height="600" alt="llama4_rope_speed_forward" src="https://github.com/user-attachments/assets/00ba8d27-8095-4427-83f7-6e2a1be4cd7f" />

backward: ✔
<img width="1000" height="600" alt="llama4_rope_speed_backward" src="https://github.com/user-attachments/assets/cce9a495-e069-4486-9e46-f0e659da459f" />

full: ✔
<img width="1000" height="600" alt="llama4_rope_speed_full" src="https://github.com/user-attachments/assets/f313f2be-67eb-4223-843a-d1da382b00a1" />

memory: ✔
<img width="1000" height="600" alt="llama4_rope_memory_full" src="https://github.com/user-attachments/assets/485e61b5-3592-437b-8143-a34ac57d84e1" />


#### 2.12 poly_norm

forward: ✔
<img width="1000" height="600" alt="poly_norm_speed_forward" src="https://github.com/user-attachments/assets/540523a2-fcdf-4cb0-b275-27854ffda130" />

backward: ✔
<img width="1000" height="600" alt="poly_norm_speed_backward" src="https://github.com/user-attachments/assets/756388a9-53dd-4e55-93d1-c33cdd2ad5a1" />

full: ✔
<img width="1000" height="600" alt="poly_norm_speed_full" src="https://github.com/user-attachments/assets/463a7a00-8d1b-4775-a838-bfe25810cf53" />

memory: ✔
<img width="1000" height="600" alt="poly_norm_memory_full" src="https://github.com/user-attachments/assets/4db0b0a0-ccf4-4fe2-8579-cde0b8a1f62e" />


#### 2.13 qwen2vl_mrope

forward: ✔
<img width="1000" height="600" alt="qwen2vl_mrope_speed_forward" src="https://github.com/user-attachments/assets/1d3fac42-0d67-4232-936f-86071c45e788" />

backward: ❌
<img width="1000" height="600" alt="qwen2vl_mrope_speed_backward" src="https://github.com/user-attachments/assets/1e0f1401-21c9-4d45-9087-d45cc1a553e8" />

full: ✔
<img width="1000" height="600" alt="qwen2vl_mrope_speed_full" src="https://github.com/user-attachments/assets/df903175-fb02-42a2-a81f-64a20cbb445f" />

memory: ✔
<img width="1000" height="600" alt="qwen2vl_mrope_memory_full" src="https://github.com/user-attachments/assets/d25ed3e5-c679-45ea-9c4c-5e196d0c094d" />


#### 2.14 tvd

forward: ✔
<img width="1000" height="600" alt="tvd_speed_forward" src="https://github.com/user-attachments/assets/83658a48-eec5-45e9-8869-106ff23a7b1e" />

backward: ✔
<img width="1000" height="600" alt="tvd_speed_backward" src="https://github.com/user-attachments/assets/9c8618d9-2f85-496b-b38e-e1b757188afa" />

full: ✔
<img width="1000" height="600" alt="tvd_speed_full" src="https://github.com/user-attachments/assets/b410a292-e739-438f-9da0-da781268b825" />

memory: ✔
<img width="1000" height="600" alt="tvd_memory_full" src="https://github.com/user-attachments/assets/1d6e0de0-ab9c-43a7-a9f3-fc223adb344a" />
