import torchvision

def batch(x, y, batch_size):
  batched_x, batched_y = [], []
  for i in range(0, len(x), batch_size):
      batched_x.append(x[i:i+batch_size])
      batched_y.append(y[i:i+batch_size])
  return batched_x, batched_y

def load_mnist(path):
    mnist_train = torchvision.datasets.MNIST(
        root=path,
        train=True,
        download=True
    )
    mnist_test = torchvision.datasets.MNIST(
        root=path,
        train=False,
        download=True
    )

    X_train, y_train = mnist_train.data.numpy(), mnist_train.targets.numpy()
    X_test, y_test = mnist_test.data.numpy(), mnist_test.targets.numpy()

    return (X_train, y_train), (X_test, y_test)
