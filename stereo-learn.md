# 双目视觉学习笔记

## 理论部分

### 双目相机模型

![扫描举例](./imges/双目相机模型.svg)

$$ \left\{
\begin{aligned}
\triangle TOP \backsim \triangle OO'D_1 \\
\triangle C_2EP \backsim \triangle C_2GD_2  \\
\end{aligned}
\right.
\Rightarrow \left\{
\begin{aligned}
\frac{x}{xl} & = \frac{z}{OO'(f)} \\
\frac{x-b}{xr} & = \frac{z}{OO'(f)}  \\
\end{aligned}
\right.
\Rightarrow \left\{
\begin{aligned}
& xl = \frac{x}{z}    \\
& xr = \frac{x-b}{z}f \\
\end{aligned}
\right.
$$
$$
\Rightarrow z = \frac{b}{xl-xr}f
$$

### 相机坐标系——像面坐标系

![扫描举例](./imges/相机坐标系—像面坐标系.svg)

$$
\begin{aligned}
\triangle PO_cA \backsim \triangle P'O_cB \Rightarrow \frac{Z_c}{y} &= \frac{PO_c}{P'O_c}=\frac{AO_c}{BO_c} \\
\triangle PO_cC \backsim \triangle P'O_cE \Rightarrow \frac{X_c}{x} &= \frac{PO_c}{P'O_c}=\frac{CO_c}{EO_c} \\
\triangle DO_cC \backsim \triangle EO_2O_c \Rightarrow \frac{Y_c}{y} &= \frac{CO_c}{EO_c}=\frac{PO_c}{P'O_c}=\frac{Z_c}{f} \\
\end{aligned}
$$

$$
\left\{
\begin{aligned}
x & = \frac{X_c}{Z_c}f  \\
x & = \frac{Y_c}{Z_c}f  \\
\end{aligned}
\right.
\Rightarrow Z_c
\begin{bmatrix} x \\ y \\ 1
\end{bmatrix}
= \begin{bmatrix} f & 0 & 0 & 0\\
0&f&0&0\\ 0&0&1&0 \end{bmatrix}
\begin{bmatrix} X_c\\Y_c\\ Z_c \\1 \end{bmatrix}
$$

$$
\left\{
\begin{aligned}
\begin{bmatrix} u \\ v \\1 \end{bmatrix} & = \begin{bmatrix} \frac{1}{\mathrm{d}x} & 0 & u_0 \\ 0 &
\frac{1}{\mathrm{d}y} & v_0 \\ 0 & 0 & 1 \end{bmatrix} \begin{bmatrix} x \\ y \\ 1 \end{bmatrix} \\
 Z_c \begin{bmatrix} x \\ y \\ 1 \end{bmatrix} & = \begin{bmatrix} f & 0 & 0 & 0\\
0&f&0&0\\ 0&0&1&0 \end{bmatrix}
\begin{bmatrix} X_c\\Y_c\\ Z_c \\1 \end{bmatrix}  \\
\end{aligned}
\right.
\Rightarrow\begin{bmatrix} u \\ v \\ 1 \end{bmatrix} =\frac{1}{Z_c}
\begin{bmatrix}\frac{1}{\mathrm{d}x} & 0 & u_0 \\ 0 &
\frac{1}{\mathrm{d}y} & v_0 \\ 0 & 0 & 1 
 \end{bmatrix}\begin{bmatrix} f & 0 & 0 & 0\\
0&f&0&0\\ 0&0&1&0 \end{bmatrix}
\begin{bmatrix} X_c\\Y_c\\ Z_c \\1 \end{bmatrix}
$$
