from agentnet.utils.format import check_list
from lasagne.layers import Layer
import numpy as np


class BroadcastLayer(Layer):
    """
    Merges certain axes of network into first (batch) axis to allow broadcasting over them.
    :param incoming: layer to be broadcasted
    :type incoming: Layer
    :param broadcasted_axes: an axis (or axes) to be broadcasted
    :type broadcasted_axes: int or tuple of int
    :force_broadcastable_batch: if True, raises an eror whenever batch (0'th) axis is not included in broadcasted_axes

    """

    def __init__(self, incoming, broadcasted_axes, force_broadcastable_batch=True, **kwargs):

        self.incoming_ndim = len(incoming.output_shape)

        # axes that are to be broadcasted -- in ascending order
        # ax%self.incoming_ndim is used to replace negative axes with N-ax+1 so that -1 becomes last axis
        self.broadcasted_axes = sorted([ax % self.incoming_ndim for ax in check_list(broadcasted_axes)])

        # sanity checks
        assert max(self.broadcasted_axes) < self.incoming_ndim
        assert len(self.broadcasted_axes) > 0
        if force_broadcastable_batch and (0 not in self.broadcasted_axes):
            raise ValueError("BroadcastLayer was asked NOT to broadcast over batch (0'th) axis.\n"
                             "If you know what you're doing, set force_broadcastable_batch=False.\n"
                             "Otherwise just add 0 to the broadcasted_axes")

        # axed that are NOT broadcasted = all other axes in respective order
        self.non_broadcasted_axes = [ax for ax in range(self.incoming_ndim) if ax not in self.broadcasted_axes]


        super(BroadcastLayer, self).__init__(incoming, **kwargs)

    def get_output_for(self, input, **kwargs):
        """
        performs theanic magic (see layer description)
        :param input: activation to be reshaped into broadcastable shape
        :param kwargs: no effect
        :return: symbolic expression for reshaped layer activation
        """

        # save symbolic input shape for unbroadcaster
        self.symbolic_input_shape = input.shape

        # dimshuffle so that the new order is [ all_broadcasted_axes, all_non_broadcasted_axes]

        input = input.dimshuffle(self.broadcasted_axes + self.non_broadcasted_axes)

        # flatten broadcasted axes into a single axis
        input = input.reshape((-1,) + tuple(input.shape[len(self.broadcasted_axes):]))

        # now shape should be [ product(broadcasted_axes_shapes), non_broadcasted_axes ]

        return input

    def get_output_shape_for(self, input_shape):

        broadcasted_shapes = [input_shape[ax] for ax in self.broadcasted_axes]

        if None not in broadcasted_shapes:
            new_batch_size = np.prod(broadcasted_shapes)
        else:
            new_batch_size = None

        non_broadcasted_shapes = tuple(input_shape[ax] for ax in self.non_broadcasted_axes)

        return (new_batch_size,) + non_broadcasted_shapes


class UnbroadcastLayer(Layer):
    """
    Does the inverse of BroadcastLayer
    :param incoming: a layer to be unbroadcasted. (!) Must have same number of dimensions as before broadcasting
    :type incoming: Layer
    :param broadcast_layer: a broadcasting to be be undone
    :type broadcast_layer: BroadcastLayer

    """

    def __init__(self, incoming, broadcast_layer, **kwargs):
        self.broadcast_layer = broadcast_layer

        #assert that dimensionality is same as before broadcast
        assert len(incoming.output_shape) == len(self.broadcast_layer.output_shape)

        super(UnbroadcastLayer, self).__init__(incoming, **kwargs)

    def get_output_for(self, input, **kwargs):
        """
        Un-broadcasts the broadcast layer (see class description)
        :param input: input tensor
        :param kwargs: no effect
        :return: un-broadcasted tensor
        """

        if not hasattr(self.broadcast_layer,"symbolic_input_shape"):
            raise ValueError("UnbroadcastLayer.get_output_for must be called after respective BroadcastLayer.get_output_for")

        # symbolic shape. dirty hack to handle "None" axes
        pre_broadcast_shape = self.broadcast_layer.symbolic_input_shape

        broadcasted_axes_shapes = tuple(pre_broadcast_shape[ax] for ax in self.broadcast_layer.broadcasted_axes)

        # convert shape from [bc_ax0*bc_ax1*.., non_bc_ax0, non_bc_ax1,...] to [bc_ax0,bc_ax1,...,non_bc_ax0,non_bc_ax1,...]
        unrolled_shape = broadcasted_axes_shapes + tuple(input.shape)[1:]
        input = input.reshape(unrolled_shape)

        # rearrange axes to their order before broadcasting
        current_dim_order = self.broadcast_layer.broadcasted_axes + self.broadcast_layer.non_broadcasted_axes

        dimshuffle_order = [current_dim_order.index(i) for i in range(len(current_dim_order))]

        return input.dimshuffle(dimshuffle_order)


    def get_output_shape_for(self, input_shape, **kwargs):

        new_non_broadcast_shapes = input_shape[1:]

        # this one is NOT symbolic. list() is used as a shallow copy op.
        original_shape = list(self.broadcast_layer.input_shape)

        # set new non-broadcasted axes shapes instead of old ones
        for ax,new_ax_shape in zip(self.broadcast_layer.non_broadcasted_axes,
                             new_non_broadcast_shapes):
            original_shape[ax] = new_ax_shape

        #return updated shape
        return tuple(original_shape)