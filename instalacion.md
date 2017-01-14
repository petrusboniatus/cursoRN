# Instalación de tensor flow en python

Pasos previos: tener linux 64 bits con python 2.7


1.Instalamos el pip
```
sudo apt-get install python-pip python-dev
```
2. Creamos una variable global para referenciar la librería
```
export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.12.1-cp27-none-linux_x86_64.whl
```

3. Instalamos tensor flow
```
sudo pip install --upgrade $TF_BINARY_UR
```
