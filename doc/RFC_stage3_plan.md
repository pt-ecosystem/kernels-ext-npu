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

#### rms_norm

forward: partially passed ❌
<img width="1000" height="600" alt="rms_norm_speed_forward" src="https://github.com/user-attachments/assets/bb50e59a-5c2b-4595-a2fd-797e1903b2c8" />

backward: partially passed ❌
<img width="1000" height="600" alt="rms_norm_speed_backward" src="https://github.com/user-attachments/assets/291e775a-45a0-4563-aa9a-4b7201a0d8f2" />

full: partially passed ❌
<img width="1000" height="600" alt="rms_norm_speed_full" src="https://github.com/user-attachments/assets/c3954975-45ac-47ad-8528-78abb46237f2" />

memory: ✔
<img width="1000" height="600" alt="rms_norm_memory_full" src="https://github.com/user-attachments/assets/0dfcb4b2-c813-4f37-8086-9aaf4c8b93cd" />


#### swiglu

forward: ✔
<img width="1000" height="600" alt="swiglu_speed_forward" src="https://github.com/user-attachments/assets/08dc0c40-132c-47b8-b661-515fc746c730" />

backward: ❌
<img width="1000" height="600" alt="swiglu_speed_backward" src="https://github.com/user-attachments/assets/2b3ca1dc-8e5e-4af0-a087-151a6309ff9d" />

full: ❌
<img width="1000" height="600" alt="swiglu_speed_full" src="https://github.com/user-attachments/assets/fc054b87-f27f-4093-8791-9e948c034012" />

memory: ✔
<img width="1000" height="600" alt="swiglu_memory_full" src="https://github.com/user-attachments/assets/ccd3f20b-4751-49cc-b8ef-f98d7eb6e778" />


#### rope

Note: The transformers version is 4.57.3. Related issue: https://github.com/linkedin/Liger-Kernel/issues/1155

forward: ✔
<img width="1000" height="600" alt="rope_speed_forward" src="https://github.com/user-attachments/assets/a9a5131d-3f89-46fe-ade4-176782f29d43" />

backward: ✔
<img width="1000" height="600" alt="rope_speed_backward" src="https://github.com/user-attachments/assets/6a5651c1-790b-433b-b0a1-39d7d365d479" />

full: ✔
<img width="1000" height="600" alt="rope_speed_full" src="https://github.com/user-attachments/assets/385f859e-80b1-41f4-b398-e85ca720f4c2" />

memory: ✔
<img width="1000" height="600" alt="rope_memory_full" src="https://github.com/user-attachments/assets/d2967186-63a0-47fc-8617-d55bf1902922" />


#### softmax

forward: ❌
<img width="1000" height="600" alt="softmax_speed_forward" src="https://github.com/user-attachments/assets/dd239a46-9206-44f3-b205-c5c40086b367" />

backward: ❌
<img width="1000" height="600" alt="softmax_speed_backward" src="https://github.com/user-attachments/assets/d874b5a3-af60-4757-8033-12e4427cd992" />

full: ❌
<img width="1000" height="600" alt="softmax_speed_full" src="https://github.com/user-attachments/assets/73bfff10-90b9-4c71-91c2-ec3ad62ebda9" />

memory: ❌
<img width="1000" height="600" alt="softmax_memory_full" src="https://github.com/user-attachments/assets/471fcb1d-42f1-482b-aaaa-0a4116164d42" />


#### group_norm

forward: ❌
<img width="1000" height="600" alt="group_norm_speed_forward" src="https://github.com/user-attachments/assets/dcd7bd26-7f19-4546-9f86-dc97fa666339" />

backward: ❌
<img width="1000" height="600" alt="group_norm_speed_backward" src="https://github.com/user-attachments/assets/6872ccf9-c4c6-440e-808a-f67410c68019" />

full: ❌
<img width="1000" height="600" alt="group_norm_speed_full" src="https://github.com/user-attachments/assets/8bb29e23-0ee0-4925-847b-751d2e36c81d" />

memory: ✔
<img width="1000" height="600" alt="group_norm_memory_full" src="https://github.com/user-attachments/assets/6e354d8f-2e46-46eb-b97c-2a25d1664a77" />


#### geglu

forward: ✔
<img width="1000" height="600" alt="geglu_speed_forward" src="https://github.com/user-attachments/assets/3a2369b6-84a3-483c-8811-27b440cffde9" />

backward: ✔
<img width="1000" height="600" alt="geglu_speed_backward" src="https://github.com/user-attachments/assets/762054ed-8a6c-4fab-aa0f-ac8584085047" />

full: ✔
<img width="1000" height="600" alt="geglu_speed_full" src="https://github.com/user-attachments/assets/cebbdba4-bb7a-414b-80f7-3beca9eec903" />

memory: ✔
<img width="1000" height="600" alt="geglu_memory_full" src="https://github.com/user-attachments/assets/9969ce18-deed-4af2-a001-955c754f3c50" />


#### cross_entropy

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


#### fused_linear_grpo_loss_sequence

forward: ❌
<img width="1000" height="600" alt="fused_linear_grpo_loss_sequence_speed_forward" src="https://github.com/user-attachments/assets/dce0a952-4689-4a3a-ad66-1c4541520590" />

backward: ✔
<img width="1000" height="600" alt="fused_linear_grpo_loss_sequence_speed_backward" src="https://github.com/user-attachments/assets/1acdf3e7-e306-407f-b3b3-47933fbd20eb" />

full: ❌
<img width="1000" height="600" alt="fused_linear_grpo_loss_sequence_speed_full" src="https://github.com/user-attachments/assets/aa44850d-9d49-4b2f-9771-207948c88616" />

memory: ❌
<img width="1000" height="600" alt="fused_linear_grpo_loss_sequence_memory_full" src="https://github.com/user-attachments/assets/dcdb4b2b-e0a8-4257-9037-105fa0121684" />


#### fused_linear_grpo_loss_token

forward: ❌
<img width="1000" height="600" alt="fused_linear_grpo_loss_token_speed_forward" src="https://github.com/user-attachments/assets/0321f85e-7c17-4216-a5eb-2f9c0ec223be" />

backward: ✔
<img width="1000" height="600" alt="fused_linear_grpo_loss_token_speed_backward" src="https://github.com/user-attachments/assets/cc52ed39-5574-4b93-af43-95f2816f680d" />

full: ❌
<img width="1000" height="600" alt="fused_linear_grpo_loss_token_speed_full" src="https://github.com/user-attachments/assets/6520ce62-2368-4e15-8d3c-10f598a81bdb" />

memory: ❌
<img width="1000" height="600" alt="fused_linear_grpo_loss_token_memory_full" src="https://github.com/user-attachments/assets/f28ec782-e046-48be-91da-e9a3adffd9be" />
