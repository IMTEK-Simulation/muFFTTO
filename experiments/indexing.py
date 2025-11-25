import numpy as np

Nx = 5
Ny = 4
I = np.arange(Nx * Ny)

# print(I)

u = np.zeros(len(I))
for j in np.arange(Ny):
    if j % 2 == 0:
        for i in np.arange(Nx):

            if i % 2 == 0:
               # print(f'i={i}',f'j={j}')

                index=i + (j * Nx)
               # print(index)
                u[index] = 1


print(u.reshape(Nx, Ny, order='F'))


##################3
I = np.arange(Nx * Ny)

# print(I)

u = np.zeros(len(I))
for j in np.arange(Ny):
    if (j+0) % 2 == 0:
        for i in np.arange(Nx):
            if (i+1) % 2 == 0:
                #print(f'i={i}',f'j={j}')

                index=i + (j * Nx)
               # print(index)
                u[index] = 1


print(u.reshape(Nx, Ny, order='F'))