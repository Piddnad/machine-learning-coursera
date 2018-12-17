function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%


% grad = 1/m * X' * (sigmoid(X*theta) - y) + lambda/m*temp

% 前向传播
X = [ones(m, 1) ,X];
a2 = sigmoid(X * Theta1');
a2 = [ones(m, 1), a2];
h = sigmoid(a2 * Theta2');

% 构造 Y 矩阵。因为现在的 y 是 5000×1 的向量，而我们想得到 5000×10 的 Y 矩阵
I = eye(num_labels);
Y = zeros(m, num_labels);                                                             
for i = 1:m,
    Y(i, :) = I(y(i), :);
end

J = -(1 / m) * sum(sum(Y .* log(h) + (1 - Y) .* log(1-h)));

% 正则化，不考虑Theta中的偏执项
temp1 = Theta1;
temp1(:, 1) = 0;
temp2 = Theta2;
temp2(:, 1) = 0;
J = J + lambda / (2 * m) * (sum(sum(temp1.^2)) + sum(sum(temp2.^2)));

delta1 = zeros(size(Theta1));
delta2 = zeros(size(Theta2));

for t = 1:m
    % Step 1 前向传播
    a1 = X(t, :)';
    z2 = Theta1 * a1;
    a2 = sigmoid(z2);
    a2 = [1; a2];
    z3 = Theta2 * a2;
    a3 = sigmoid(z3);

    % Step 2 计算sigma3
    err3 = zeros(num_labels, 1);
    for k = 1:num_labels
        err3(k) = a3(k) - (y(t) == k);
    end

    % step 3
    err2 = Theta2' * err3;                % err_2有26行！！！
    err2 = err2(2:end) .* sigmoidGradient(z2);   % 去掉第一个误差值，减少为25. sigmoidGradient(z_2)只有25行！！！
    
    % step 4
    delta2 = delta2 + err3 * a2';
    delta1 = delta1 + err2 * a1';
end

% step 5
temp1 = Theta1;
temp1(:, 1) = 0;
temp2 = Theta2;
temp2(:, 1) = 0;
Theta1_grad = 1 / m * delta1 + lambda/m * temp1;
Theta2_grad = 1 / m * delta2 + lambda/m * temp2;

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
