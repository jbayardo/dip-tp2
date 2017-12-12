import matplotlib.pyplot as plt
import numpy as np
import skimage.measure as skm
import skimage as ski
import skimage.io
import itertools
from scipy import fftpack as ffp


class HuffmanTree(object):
    class HuffmanNode(object):
        def __init__(self, left, right, weight=None):
            self.left = left
            self.right = right
            self.weight = weight

        def __lt__(self, other):
            return self.weight < other.weight

        _is_leaf = False

        def is_leaf(self):
            return self._is_leaf

    class HuffmanLeaf(HuffmanNode):
        def __init__(self, symbol, weight=None):
            super().__init__(None, None, weight)
            self.symbol = symbol

        _is_leaf = True

    def __init__(self, head):
        self._head = head
        self._encoding_table = self.build_encoding_table()

    def build_encoding_table(self):
        encoding_table = {}

        if self._head.is_leaf():
            return None

        queue = [(self._head, '')]
        while len(queue) > 0:
            node, prefix = queue.pop()
            if node.is_leaf():
                # We need to invert the prefix because the first operation might be 'left', in which case it would be
                # lost if we had more bits of accuracy than strictly needed
                prefix = int(prefix[::-1], 2)
                encoding_table[node.symbol] = prefix
            else:
                queue.append((node.left, prefix + '0'))
                queue.append((node.right, prefix + '1'))

        return encoding_table

    @classmethod
    def from_run_length_encoding(cls, encoding):
        weights = {}
        for (weight, symbol) in encoding:
            weights[symbol] = weights.get(symbol, 0) + weight
        return cls.from_weight_dictionary(weights)

    @classmethod
    def from_weight_dictionary(cls, weights):
        from heapq import heappush, heappop, heapify

        queue = [cls.HuffmanLeaf(symbol, weights[symbol]) for symbol in weights]
        heapify(queue)
        while len(queue) > 1:
            left = heappop(queue)
            right = heappop(queue)
            node = cls.HuffmanNode(left, right, left.weight + right.weight)
            heappush(queue, node)

        return HuffmanTree(queue[0])

    def into_weight_dictionary(self):
        weights = {}

        queue = [self._head]
        while len(queue) > 0:
            node = queue.pop()

            if node.is_leaf():
                weights[node.symbol] = node.weight
            else:
                queue.append(node.left)
                queue.append(node.right)

        return weights

    def encode(self, symbol):
        if self._head.is_leaf():
            return 0
        return self._encoding_table[symbol]

    def encode_many(self, symbols):
        if self._head.is_leaf():
            return [0] * len(symbols)

        return [self._encoding_table[symbol] for symbol in symbols]

    def decode_many(self, codes):
        if self._head.is_leaf():
            return [self._head.symbol] * len(codes)
        return [self._decode_non_leaf(code) for code in codes]

    def decode(self, code):
        if self._head.is_leaf():
            return self._head.symbol

        return self._decode_non_leaf(code)

    def _decode_non_leaf(self, code):
        node = self._head
        while code > 0:
            if code & 1 == 0:
                node = node.left
            else:
                node = node.right

            code = code >> 1
        while not node.is_leaf():
            node = node.left
        return node.symbol


class ImageProcessor(object):
    def apply(self, image):
        raise NotImplementedError()


class SquareChunkProcessor(ImageProcessor):
    def __init__(self, chunk_size=8):
        self.image = None
        self.height = None
        self.width = None
        self.chunk_size = chunk_size

    def apply(self, image):
        self.image = image
        # TODO: hay que hacer que self._processed_image tenga el mismo tamanio que la nueva imagen
        # self.image = np.pad(image,
        #                     pad_width=[
        #                         (0, self.chunk_size - (image.shape[0] % self.chunk_size)),
        #                         (0, self.chunk_size - (image.shape[1] % self.chunk_size))],
        #                     mode='constant')
        self.height = self.image.shape[0]
        self.width = self.image.shape[1]

        height_blocks = self.height // self.chunk_size
        width_blocks = self.width // self.chunk_size

        for height_block_index in range(height_blocks):
            for width_block_index in range(width_blocks):
                self._process_chunk(height_block_index, width_block_index)

    def _process_chunk(self, height_block_index, width_block_index):
        raise NotImplementedError()


class Codec(SquareChunkProcessor):
    def __init__(self, chunk_size=8):
        super().__init__(chunk_size)
        self._processed_image = None

    def _process_chunk(self, height_block_index, width_block_index):
        height_block_start = height_block_index * self.chunk_size
        height_block_end = (height_block_index + 1) * self.chunk_size
        width_block_start = width_block_index * self.chunk_size
        width_block_end = (width_block_index + 1) * self.chunk_size

        chunk = self.image[height_block_start:height_block_end, width_block_start:width_block_end]
        chunk = self._pipeline(height_block_index, width_block_index, chunk)
        self._processed_image[height_block_start:height_block_end, width_block_start:width_block_end] = chunk

    def _pipeline(self, height_block_index, width_block_index, chunk):
        chunk = self._transform(chunk)
        chunk = self._quantize(chunk)
        return chunk

    def _quantize(self, chunk):
        raise NotImplementedError()

    def _transform(self, chunk):
        raise NotImplementedError()


class GreyscaleEncoder(Codec):
    @staticmethod
    def _into_int_range(image):
        image = image.astype(np.int16, copy=False)
        image = image - np.power(2, 7) * np.ones_like(image)
        image = image.astype(np.int8, copy=False)
        return image

    def apply(self, image):
        # TODO: assert image.dtype == np.uint8

        image = self._into_int_range(image)
        self._processed_image = np.zeros_like(image, dtype=np.double)
        super().apply(image)

        return self._processed_image

    def _quantize(self, chunk):
        raise NotImplementedError()

    def _transform(self, chunk):
        return ffp.dct(ffp.dct(chunk.T, norm='ortho').T, norm='ortho')


class GreyscaleDecoder(Codec):
    @staticmethod
    def _into_uint_range(image):
        image = image + np.power(2, 7) * np.ones_like(image)
        image = image.astype(np.uint8, copy=False)
        return image

    def apply(self, image):
        assert image.dtype == np.uint8

        self._processed_image = np.zeros_like(image)
        super().apply(image)
        self._processed_image = self._into_uint_range(self._processed_image)
        return self._processed_image

    def _quantize(self, chunk):
        raise NotImplementedError()

    def _transform(self, chunk):
        return ffp.idct(ffp.idct(chunk.T, norm='ortho').T, norm='ortho')


class TableGreyscaleEncoder(GreyscaleEncoder):
    def __init__(self, quant_table, quant_threshold=None, chunk_size=8):
        super().__init__(chunk_size)
        self._quant_table = quant_table
        self._quant_threshold = quant_threshold

    def _quantize(self, chunk):
        chunk = np.divide(chunk, self._quant_table)
        chunk = np.round(chunk)

        if self._quant_threshold is not None:
            chunk[chunk <= self._quant_threshold] = 0.0

        return chunk


class TableGreyscaleDecoder(GreyscaleDecoder):
    def __init__(self, quant_table, chunk_size=8):
        super().__init__(chunk_size)
        self._quant_table = quant_table

    def _quantize(self, chunk):
        return np.multiply(chunk, self._quant_table)


def zigzag(n):
    """
    Produce a list of indexes that traverse a matrix of size n * n using the JPEG zig-zag order.
    Taken from https://rosettacode.org/wiki/Zig-zag_matrix#Alternative_version.2C_Translation_of:_Common_Lisp
    
    :param n: size of square matrix to iterate over
    :return: list of indexes in the matrix, sorted by visit order
    """
    def move(i, j):
        if j < (n - 1):
            return max(0, i - 1), j + 1
        else:
            return i + 1, j

    mask = []
    x, y = 0, 0
    for v in range(n * n):
        mask.append((y, x))
        # Inverse: mask[y][x] = v

        if (x + y) & 1:
            x, y = move(x, y)
        else:
            y, x = move(y, x)

    return mask


class DecompressingTableGreyscaleDecoder(TableGreyscaleDecoder):
    def __init__(self, decompressor, quant_table, chunk_size=8):
        super().__init__(quant_table, chunk_size)
        self._zigzag_order = zigzag(chunk_size)
        self._decompressor = decompressor

    def apply(self, image):
        image = self._decompress(image)
        super().apply(image)
        return self._processed_image

    def _decompress(self, file):
        width = file['width']
        height = file['height']
        image = np.zeros(shape=(width, height), dtype=np.int16)

        assert width % self.chunk_size == 0
        assert height % self.chunk_size == 0

        height_blocks = height // self.chunk_size
        width_blocks = width // self.chunk_size

        last_chunk_dc_coefficient = 0
        for height_block_index in range(height_blocks):
            for width_block_index in range(width_blocks):
                index = (height_block_index, width_block_index)
                height_block_start = height_block_index * self.chunk_size
                height_block_end = (height_block_index + 1) * self.chunk_size
                width_block_start = width_block_index * self.chunk_size
                width_block_end = (width_block_index + 1) * self.chunk_size

                dc_coefficient = file['DC'][index] + last_chunk_dc_coefficient
                last_chunk_dc_coefficient = dc_coefficient

                compressed_chunk = list(zip(*file['chunks'][index]))
                lengths_of_runs, codes = compressed_chunk[0], compressed_chunk[1]
                symbols = self._decompressor.decode_many(codes)

                decompressed_chunk = self._run_length_decode(dc_coefficient, lengths_of_runs, symbols)
                image[height_block_start:height_block_end, width_block_start:width_block_end] = decompressed_chunk

        return image

    def _run_length_decode(self, dc_coefficient, lengths_of_runs, symbols):
        chunk = np.zeros(shape=(self.chunk_size, self.chunk_size))
        chunk[0, 0] = dc_coefficient
        position = itertools.islice(self._zigzag_order, 1, None)
        for length, symbol in zip(lengths_of_runs, symbols):
            while length > 0:
                i, j = next(position)
                chunk[i, j] = symbol
                length -= 1
        return chunk


class CompressingTableGreyscaleEncoder(TableGreyscaleEncoder):
    def __init__(self, compressor, quant_table, quant_threshold=None, chunk_size=8):
        super().__init__(quant_table, quant_threshold, chunk_size)
        self._zigzag_order = zigzag(chunk_size)
        self._compressor = compressor
        self._compressed_image = None
        self._last_chunk_dc_coefficient = None

    def apply(self, image):
        self._last_chunk_dc_coefficient = 0
        self._compressed_image = {
            'DC': {},
            'chunks': {}
        }

        super().apply(image)

        self._compressed_image['width'] = self.width
        self._compressed_image['height'] = self.height
        return self._compressed_image

    def _pipeline(self, height_block_index, width_block_index, chunk):
        chunk = super()._pipeline(height_block_index, width_block_index, chunk)
        # Add a simple compression step into the pipeline
        self._compress(height_block_index, width_block_index, chunk)
        return chunk

    def _compress(self, height_block_index, width_block_index, chunk):
        index = (height_block_index, width_block_index)

        # Run-length encode the chunk
        lengths_of_runs, symbols = self._run_length_encode(chunk)

        # Compress the chunk
        compressed_chunk = self._compressor.encode_many(symbols)

        # Add to compressed file
        self._compressed_image['DC'][index] = chunk[0, 0] - self._last_chunk_dc_coefficient
        self._last_chunk_dc_coefficient = chunk[0, 0]

        self._compressed_image['chunks'][index] = list(zip(lengths_of_runs, compressed_chunk))

    def _run_length_encode(self, chunk):
        lengths_of_runs = []
        symbols = []

        # The first entry is the DC, which is encoded differently, so we skip it
        symbol = chunk[0, 1]
        length_of_run = 1
        for (i, j) in itertools.islice(self._zigzag_order, 2, None):
            if chunk[i, j] != symbol:
                lengths_of_runs.append(length_of_run)
                symbols.append(symbol)

                symbol = chunk[i, j]
                length_of_run = 0

            length_of_run += 1
        lengths_of_runs.append(length_of_run)
        symbols.append(symbol)

        return lengths_of_runs, symbols


class YCbCrSubsamplingEncoder(ImageProcessor):
    def __init__(self, processors, sampling):
        assert 'Y' in processors
        assert 'Cb' in processors
        assert 'Cr' in processors

        self._processors = processors
        self._sampling = sampling

        self.height = None
        self.width = None
        self.image = None

    @staticmethod
    def rgb2ycbcr(im):
        xform = np.array([[.299, .587, .114], [-.1687, -.3313, .5], [.5, -.4187, -.0813]])
        ycbcr = im.dot(xform.T)
        ycbcr[:, :, [1, 2]] += 128
        return np.uint8(ycbcr)

    @staticmethod
    def ycbcr2rgb(im):
        xform = np.array([[1, 0, 1.402], [1, -0.34414, -.71414], [1, 1.772, 0]])
        rgb = im.astype(np.float)
        rgb[:, :, [1, 2]] -= 128
        return np.uint8(rgb.dot(xform.T))

    def chroma_subsample(self, subchannel, subsampling):
        return skm.block_reduce(subchannel,
                                block_size=(subsampling[0], subsampling[1]),
                                func=np.mean)

    def apply(self, image):
        channels = image.shape[2]

        assert 3 <= channels <= 4
        assert channels == 3 or 'A' in self._processors

        self.image = image
        self.height = self.image.shape[0]
        self.width = self.image.shape[1]

        # Alpha subsampling
        A_subsampled = None
        if channels == 4:
            A_subsampled = self.chroma_subsample(self.image[:,:,3],
                                                 self._sampling)
            A_subsampled = self._processors['A'].apply(A_subsampled)

        ycbcr = YCbCrSubsamplingEncoder.rgb2ycbcr(self.image[:, :, :3])

        Y = ycbcr[:,:,0]
        Y = self._processors['Y'].apply(Y)

        Cb_subsampled = self.chroma_subsample(ycbcr[:,:,1],
                                              self._sampling)
        Cb_subsampled = self._processors['Cb'].apply(Cb_subsampled)

        Cr_subsampled = self.chroma_subsample(ycbcr[:,:,2],
                                              self._sampling)
        Cr_subsampled = self._processors['Cr'].apply(Cr_subsampled)

        return {
            'height': self.height,
            'width': self.width,
            'sampling': self._sampling,
            'Y': Y,
            'Cb': Cb_subsampled,
            'Cr': Cr_subsampled,
            'A': A_subsampled
        }


if __name__ == '__main__':
    table = np.array([[17, 18, 24, 47, 99, 128, 192, 256],
                      [18, 21, 26, 66, 99, 192, 256, 512],
                      [24, 26, 56, 99, 128, 256, 512, 512],
                      [47, 66, 99, 128, 256, 512, 1024, 1024],
                      [99, 99, 128, 256, 512, 1024, 2048, 2048],
                      [128, 192, 256, 512, 1024, 2048, 4096, 4096],
                      [192, 256, 512, 1024, 2048, 4096, 8192, 8192],
                      [256, 512, 512, 1024, 2048, 4096, 8192, 8192]])

    s = 7
    table[0:s, 0:s] = np.ones_like(table[0:s, 0:s])

    table = np.array([[16, 11, 10, 16, 24, 40, 51, 61],
                      [12, 12, 14, 19, 26, 58, 60, 55],
                      [14, 13, 16, 24, 40, 57, 69, 56],
                      [14, 17, 22, 29, 51, 87, 80, 62],
                      [18, 22, 37, 56, 68, 109, 103, 77],
                      [24, 35, 55, 64, 81, 104, 113, 92],
                      [49, 64, 78, 87, 103, 121, 120, 101],
                      [72, 92, 95, 98, 112, 100, 103, 99]])

    table = 1 * np.ones_like(table)

    # TODO: esto tendria que ser un buen compresor, hay que usar una tabla copada. Esta tabla flashea fuerte.
    translator = HuffmanTree.from_weight_dictionary({4: 54})

    encoder = CompressingTableGreyscaleEncoder(translator, table)
    decoder = DecompressingTableGreyscaleDecoder(translator, table)

    color = ski.io.imread("D:\\jbayardo\\Documents\\dip-tp1\\data\\color\\1908iv.png")
    rgb_encoder = YCbCrSubsamplingEncoder({
        'Y': encoder,
        'Cb': encoder,
        'Cr': encoder,
        'A': encoder
    }, (2, 2))
    rgb_decoded = rgb_encoder.apply(color)
    pass

    # lena = ski.io.imread("D:\\jbayardo\\Documents\\dip-tp1\\data\\test\\barbara.png")
    # encoded, compressed = encoder.apply(lena)
    # decoded = decoder.apply(compressed)
    # ski.io.imshow(decoded)
    # plt.show()

