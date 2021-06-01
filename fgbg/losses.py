from torch.nn.modules.loss import NLLLoss


class WeightedBinaryCrossEntropyLoss(NLLLoss):
    r"""The WeightedBinaryCrossEntropyLoss loss. It is useful to train a binary output maps.
    Documentation and implementation based on pytorch BCEWithLogitsLoss.

    If provided, the optional argument :attr:`weight` will balance the 1's
    with respect to the 0's:
    The weight is recommended to be the ratio of 0's in an image.
    Often 90% of the target binary maps is 0 while only 10% is 1.
    Having beta = 0.9 scales the losses on the target-1 pixels with 0.9
    and the losses on target-0 pixels with 0.1.

    The `input` given through a forward call is expected to contain
    probabilities for each pixel. `input` has to be a Tensor of size
    :math:`(minibatch, d_1, d_2, ..., d_K)`
    with :math:`K \geq 1` for the `K`-dimensional case (described later).

    The `target` that this loss expects should be a class index in the range :math:`[1]`
    The loss can be described as:

    .. math::
        \ell(x, y) = L = \{l_1,\dots,l_N\}^\top, \quad
        l_n = - \beta * \sum_{(j \in Y^+)} log(Pr(x_j = 1)) -
        (1-\beta) \sum_{(j \in Y^-)} log(Pr(x_j = 0))

    where :math:`x` is the probability input, :math:`y` is the target,
    :math:`\beta` is the balancing weight, and
    :math:`N` is the batch size.

    If :attr:`reduction` is not ``'none'``
    (default ``'mean'``), then

    .. math::
        \ell(x, y) = \begin{cases}
            \sum_{n=1}^N \frac{1}{\sum_{n=1}^N w_{y_n}} l_n, &
            \text{if reduction} = \text{'mean';}\\
            \sum_{n=1}^N l_n,  &
            \text{if reduction} = \text{'sum'.}
        \end{cases}

    Args:
        beta (Tensor, optional): a manual rescaling weight given to each
            class. If given, it has to be a Tensor of size `C`. Otherwise, it is
            treated as if having all ones.
        reduction (string, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``:
            no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be summed.
            Note: :attr:`size_average`
            and :attr:`reduce` are in the process of being deprecated,
            and in the meantime,
            specifying either of those two args will override :attr:`reduction`.
            Default: ``'mean'``

    Shape:
        - Input: :math:`(N, C)` where `C = number of classes`, or
          :math:`(N, C, d_1, d_2, ..., d_K)` with :math:`K \geq 1`
          in the case of `K`-dimensional loss.
        - Target: :math:`(N)` where each value is :math:`0 \leq \text{targets}[i]
        \leq C-1`, or
          :math:`(N, d_1, d_2, ..., d_K)` with :math:`K \geq 1` in the case of
          K-dimensional loss.
        - Output: scalar.
          If :attr:`reduction` is ``'none'``, then the same size as the target:
          :math:`(N)`, or
          :math:`(N, d_1, d_2, ..., d_K)` with :math:`K \geq 1` in the case
          of K-dimensional loss.
    """

    def __init__(self, beta=0.5, reduction="mean"):
        super(NLLLoss, self).__init__()
        self._beta = beta
        assert 0 <= self._beta <= 1
        self._reduction = reduction

    def forward(self, inputs, targets):
        assert targets.max() <= 1, FloatingPointError(f"got target max {targets.max()}")
        assert targets.min() >= 0, FloatingPointError(f"got target min {targets.min()}")
        assert inputs.max() <= 1, FloatingPointError(f"got inputs max {inputs.max()}")
        assert inputs.min() >= 0, FloatingPointError(f"got inputs min {inputs.min()}")
        dimension = len(inputs.shape)

        # if an input value == 0, the log value is -inf, where a -1 * -inf == nan.
        epsilon = 1e-3
        unreduced_loss = (
            -self._beta * targets * (inputs + epsilon).log()
            - (1 - self._beta) * (1 - targets) * (1 - inputs + epsilon).log()
        )
        # average over all dimensions except the batch dimension
        unreduced_loss = unreduced_loss.mean(
            dim=tuple([i + 1 for i in range(dimension - 1)])
        )
        if self._reduction == "none":
            return unreduced_loss
        elif self._reduction == "mean":
            return unreduced_loss.mean()
        elif self._reduction == "sum":
            return unreduced_loss.sum()
        else:
            raise NotImplementedError


class DeepSupervisedWeightedBinaryCrossEntropyLoss(WeightedBinaryCrossEntropyLoss):
    def __init__(self, beta=0.5, mode: str = 'deep_supervision'):
        super().__init__(beta=beta, reduction="mean")
        self.mode = mode

    def forward(self, inputs, target):
        loss = 0
        if self.mode == 'deep_supervision':
            # loop over inputs with target and apply WeightedBinaryCrossEntropyLoss
            for k in ['prob1', 'prob2', 'prob3', 'prob4', 'final_prob']:
                loss += 1 / 5. * super().forward(inputs[k], target)
        else:  # mode indicates key from inputs to use
            loss += super().forward(inputs[self.mode])
        return loss
