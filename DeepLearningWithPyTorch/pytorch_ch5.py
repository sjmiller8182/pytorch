import torch
from torch import optim

# create a linear dataset
t_c = [0.5, 14, 15, 28, 11, 8, 3, -4, 6, 13, 21]
t_u = [
    35.7, 55.9, 58.2, 81.9, 56.3, 48.9,
     33.9, 21.8, 48.4, 60.4, 68.4
     ]

# create tensors from the lists
t_c = torch.tensor(t_c)
t_u = torch.tensor(t_u)

# make the inputs a bit smaller
t_un = t_u * 0.1

# represent the model
def model (t_u, w, b):
    return w * t_u + b

def loss_fn(t_p, t_c):
    squared_diffs = (t_p - t_c)**2
    return squared_diffs.mean()

w = torch.ones(())
b = torch.zeros(())

def training_loop(n_epochs, optimizer, params, t_u, t_c):
    for epoch in range(1, n_epochs + 1):
        
        t_p = model(t_u, *params)
        loss = loss_fn(t_p, t_c)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 500 == 0:
            print(f"Epoch {epoch}, Loss {loss}")
    return params

params = torch.tensor([1., 0.], requires_grad=True)
# create SGD optim
optimizer = optim.SGD([params], lr=1e-2)

training_loop(
    5000,
    optimizer,
    params,
    t_un,
    t_c
)

# now lets do a neural network

n_sample = t_u.shape[0]
n_val = int(n_sample * 0.2)

shuffled_indicies = torch.randperm(n_sample)

train_indicies = shuffled_indicies[:-n_val]
val_indicies = shuffled_indicies[-n_val:]

train_t_u = t_u[train_indicies] * 0.1
train_t_c = t_c[train_indicies]

val_t_u = t_u[val_indicies] * 0.1
val_t_c = t_c[val_indicies]

print(train_indicies)
print(val_indicies)

def training_loop(
    n_epochs, optimizer, params,
    train_t_u, train_t_c,
    val_t_u, val_t_c
):
    for epoch in range(1, n_epochs + 1):
        
        train_t_p = model(train_t_u, *params)
        train_loss = loss_fn(train_t_p, train_t_c)

        with torch.no_grad():
            val_t_p = model(val_t_u, *params)   
            val_loss = loss_fn(val_t_p, val_t_c)

        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        if epoch <= 3 or epoch % 500 == 0:
            print(
                f"Epoch {epoch}, Loss {train_loss:.4f},"
                f" Validation Loss {val_loss:.4f}"
                )
    return params

params = torch.tensor([1., 0.], requires_grad=True)
# create SGD optim
optimizer = optim.SGD([params], lr=1e-2)

training_loop(
    5000,
    optimizer,
    params,
    train_t_u, train_t_c,
    val_t_u, val_t_c
)
