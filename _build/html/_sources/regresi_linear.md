# Regresi Linear

## Analisis Data Menggunakan Regresi Linear

Data : 
A = (2,2)
B = (4,3)
C = (5,5)
D = (3,4)
E = (3,3)
F = (4,5)
G = (5,6)

### 1. Menghitung Nilai Koefisien
menghitung nilai koefisien intersep ($\beta_0$) dan kemiringan ($\beta_1$) menggunakan rumus persamaan normal:
![rumusregresi1](https://hackmd.io/_uploads/SyuBHqckze.png)

###### 1. Menyusun Matriks Awal ($X$ dan $Y$)
Sesuai aturan perkalian matriks untuk regresi, kita menambahkan kolom dummy berupa angka 1 di sebelah kiri nilai $X$ asli untuk mengakomodasi nilai intersep ($\beta_0$).
$$X = \begin{bmatrix} 1 & 2 \\ 1 & 4 \\ 1 & 5 \\ 1 & 3 \\ 1 & 3 \\ 1 & 4 \\ 1 & 5 \end{bmatrix}, \quad Y = \begin{bmatrix} 2 \\ 3 \\ 5 \\ 4 \\ 3 \\ 5 \\ 6 \end{bmatrix}$$

###### 2. Transpose Matriks ($X^T$)\
Membalik baris matriks $X$ menjadi kolom:
$$X^T = \begin{bmatrix} 1 & 1 & 1 & 1 & 1 & 1 & 1 \\ 2 & 4 & 5 & 3 & 3 & 4 & 5 \end{bmatrix}$

###### 3. Menghitung Perkalian Matriks ($X^T X$)
Kalikan matriks $X^T$ (ukuran $2 \times 7$) dengan matriks $X$ (ukuran $7 \times 2$):
$$X^T X = \begin{bmatrix} 1+1+1+1+1+1+1 & 2+4+5+3+3+4+5 \\ 2+4+5+3+3+4+5 & 2^2+4^2+5^2+3^2+3^2+4^2+5^2 \end{bmatrix}$$

$$X^T X = \begin{bmatrix} 7 & 26 \\ 26 & 104 \end{bmatrix}$$

###### 4. Menghitung Invers Matriks $(X^T X)^{-1}$
Untuk mencari invers dari matriks $2 \times 2$, rumusnya adalah menukar posisi elemen diagonal utama, memberi tanda negatif pada diagonal sekunder, lalu dibagi dengan determinan.
- Hitung Determinan (det):
$$\text{det} = (7 \times 104) - (26 \times 26)$$
  $$\text{det} = 728 - 676 = 52$$
- Hitung Invers:
$$(X^T X)^{-1} = \frac{1}{52} \begin{bmatrix} 104 & -26 \\ -26 & 7 \end{bmatrix} = \begin{bmatrix} \frac{104}{52} & -\frac{26}{52} \\ -\frac{26}{52} & \frac{7}{52} \end{bmatrix} = \begin{bmatrix} 2 & -0.5 \\ -0.5 & \frac{7}{52} \end{bmatrix}$$

5. Menghitung Perkalian Matriks ($X^T Y$)kalikan matriks $X^T$ dengan vektor target $Y$:
$$X^T Y = \begin{bmatrix} 2+3+5+4+3+5+6 \\ (2\times2)+(4\times3)+(5\times5)+(3\times4)+(3\times3)+(4\times5)+(5\times6) \end{bmatrix}$$
$$X^T Y = \begin{bmatrix} 28 \\ 4+12+25+12+9+20+30 \end{bmatrix} = \begin{bmatrix} 28 \\ 112 \end{bmatrix}$$

###### 6. Menghitung Nilai Koefisien $\hat{\beta}$
Sekarang kita kalikan hasil invers pada Langkah 4 dengan hasil perkalian pada Langkah 5:
$$\hat{\beta} = \begin{bmatrix} 2 & -0.5 \\ -0.5 & \frac{7}{52} \end{bmatrix} \begin{bmatrix} 28 \\ 112 \end{bmatrix}$$
- Mencari Intercept ($\beta_0$):
$$\beta_0 = (2 \times 28) + (-0.5 \times 112) = 56 - 56 = \mathbf{0}$$
- Mencari Slope ($\beta_1$):
$$\beta_1 = (-0.5 \times 28) + \left(\frac{7}{52} \times 112\right) = -14 + \frac{784}{52} = -14 + 15.0769 = \mathbf{1.0769}$$

###### 7. Hasil Persamaan Garis Regresi Linier
Berdasarkan perhitungan analitik dari data titik GeoGebra Anda, didapatkan model persamaan garis lurus terbaiknya adalah:
$$Y = 0 + 1.0769 \cdot X \quad \Rightarrow \quad Y = 1.0769 \cdot X$$


### 2. Program menghitung koefisien regresi dengan  libarary dari sklearn from sklearn.linear_model import LinearRegression:
```
import numpy as np
from sklearn.linear_model import LinearRegression

X = np.array([[2], [4], [5], [3], [3], [4], [5]])
Y = np.array([2, 3, 5, 4, 3, 5, 6])

model = LinearRegression()
model.fit(X, Y)

b0_sklearn = model.intercept_  
b1_sklearn = model.coef_[0]    

print(" METODE SCIKIT-LEARN ")
print(f"Intercept (b0) : {b0_sklearn:.4f}")
print(f"Slope (b1)     : {b1_sklearn:.4f}")
print(f"Persamaan      : Y = {b0_sklearn:.4f} + {b1_sklearn:.4f} * X\n")
```
#### Output Program
```
METODE SCIKIT-LEARN
Intercept (b0) : -0.0000
Slope (b1)     : 1.0769
Persamaan      : Y = -0.0000 + 1.0769 * X
```

#### Metode Analitik Matriks
```

# 1. Menambahkan kolom bias (angka 1) ke matriks X
X_bias = np.hstack((np.ones((X.shape[0], 1)), X))

# 2. Menghitung X Transpose
X_T = X_bias.T

# 3. Menghitung (X^T * X)
X_T_X = np.dot(X_T, X_bias)

# 4. Menghitung Invers dari (X^T * X)
X_T_X_inv = np.linalg.inv(X_T_X)

# 5. Menghitung X^T * Y
X_T_Y = np.dot(X_T, Y)

# 6. Menghitung Nilai Koefisien Beta (β)
beta = np.dot(X_T_X_inv, X_T_Y)

b0_analitik = beta[0]
b1_analitik = beta[1]

print(f"Intercept (b0) : {b0_analitik:.4f}")
print(f"Slope (b1)     : {b1_analitik:.4f}")
print(f"Persamaan      : Y = {b0_analitik:.4f} + {b1_analitik:.4f} * X")
```
#### Output Program
```
Intercept (b0) : 0.0000
Slope (b1)     : 1.0769
Persamaan      : Y = 0.0000 + 1.0769 * X
```





