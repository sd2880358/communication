# 12/6
## method 1 (constrain noise):

- total_s_loss = identity_s + total_gen_loss
- total_n_loss = identity_n + total_gen_loss + n_loss
- total_i_loss = identity_g_loss + total_gen_loss
### Result in 500 epochs:
- total_loss = 0.11
- signal_loss = 3.92
- noise_loss = 0.29

## method 2 (without constrain noise):

- total_s_loss = identity_s + total_gen_loss + identity_g_loss
- total_n_loss = total_gen_loss + identity_g_loss
- total_i_loss = identity_g_loss + total_gen_loss

### Result in 500 epochs:

- total_loss = 0.11
- signal_loss = 3.91
- noise_loss = 2.12 

## method 3 (without constrain noise):
- total_s_loss = identity_s + total_gen_loss
- total_n_loss = total_gen_loss + identity_g_loss
- total_i_loss = identity_g_loss + total_gen_loss