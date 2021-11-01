# Pytorch_Lightning



With Lightning, most users don’t have to think about when to call `.zero_grad()`, `.backward()` and `.step()` since Lightning automates that for you.

Under the hood, Lightning does the following:

```python
for epoch in epochs:
    for batch in data:

        def closure():
            loss = model.training_step(batch, batch_idx, ...)
            optimizer.zero_grad()
            loss.backward()
            return loss

        optimizer.step(closure)

    for lr_scheduler in lr_schedulers:
        lr_scheduler.step()
```



In the case of **multiple** optimizers, Lightning does the following:

```python
for epoch in epochs:
    for batch in data:
        for opt in optimizers:

            def closure():
                loss = model.training_step(batch, batch_idx, optimizer_idx)
                opt.zero_grad()
                loss.backward()
                return loss

            opt.step(closure)

    for lr_scheduler in lr_schedulers:
        lr_scheduler.step()
```



As can be seen in the code snippet above, Lightning defines a closure with `training_step`, `zero_grad` and `backward` for the optimizer to execute. 





## Optimization



### Manual optimization









### Automatic optimization