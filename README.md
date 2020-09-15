# Improved-registration-method
Combined with image pyramid, continuous image representation of cubic B-spline curve and Parzen histogram estimation

概率密度分布使用Parzen直方图。由于固定图像概率密度函数对于导数不起作用，不需要平滑。因此，使用零阶BSpline内核为固定图像强度PDF。另一方面，要确保平滑度，使用三阶BSpline内。
