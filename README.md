Tensorflow Linear SVM
===

A demonstration of how you can use [TensorFlow](http://www.tensorflow.org/) to
implement a L2-norm support vector machine (SVM) in primal form. 

`linear_svm.py` optimizes the following SVM cost using gradient descent: 

<img
src="http://www.sciweavers.org/tex2img.php?eq=L%28w%3BD%29%20%3D%20%5Cfrac%7B1%7D%7B2%7D%7C%7Cw%7C%7C_2%5E2%20%2B%20C%5Csum_%7Bi%3D1%7D%5EN%20%5Cmathrm%7Bmax%7D%280%2C1-y_i%28w%5ETx_i%20%2B%20b%29%29&bc=White&fc=Black&im=png&fs=18&ff=modern&edit=0"
align="center" border="0" alt="L(w;D) = \frac{1}{2}||w||_2^2 + C\sum_{i=1}^N
\mathrm{max}(0,1-y_i(w^Tx_i + b))" width="550" height="75" />

where 

<img
src="http://www.sciweavers.org/tex2img.php?eq=D%20%3D%20%5C%7B%28x_i%2Cy_i%29%5C%7D_%7Bi%3D1%7D%5EN%2C%5C%3Bx_i%20%5Cin%20R%5Ed%5C%3B%5Cmathrm%7Band%7D%5C%3By_i%20%5Cin%20%5C%7B-1%2C%2B1%5C%7D&bc=White&fc=Black&im=png&fs=18&ff=modern&edit=0"
align="center" border="0" alt="D = \{(x_i,y_i)\}_{i=1}^N,\;x_i \in
R^d\;\mathrm{and}\;y_i \in \{-1,+1\}" width="485" height="31" />.

The first part of the cost function, i.e. the regularization part, is
implemented by the `regularization_loss` expression, and the second part is
implemented by the `hinge_loss` expression in the code. 

On a linearly separable, 2D data, the code gives the following decision
boundary: 


<img src="result.png" width="500px"/>

The code here is inspired by the repository
[try-tf](https://github.com/jasonbaldridge/try-tf).
