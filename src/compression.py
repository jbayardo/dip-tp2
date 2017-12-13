import itertools
import os
import pickle

import numpy as np
import skimage as ski
import skimage.io
import skimage.measure as skm
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
        assert chunk_size > 0
        self.image = None
        self.height = None
        self.width = None
        self.chunk_size = chunk_size

    def apply(self, image):
        self.image = image
        self.height = self.image.shape[0]
        self.width = self.image.shape[1]

        height_blocks = self.height // self.chunk_size
        if self.height % self.chunk_size != 0:
            height_blocks += 1

        width_blocks = self.width // self.chunk_size
        if self.width % self.chunk_size != 0:
            width_blocks += 1

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
        assert image.dtype == np.uint8

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
        assert quant_table is not None
        assert quant_threshold is None or quant_threshold > 0
        super().__init__(chunk_size)
        self._quant_table = quant_table
        self._quant_threshold = quant_threshold

    def _quantize(self, chunk):
        chunk = np.divide(chunk, self._quant_table)
        chunk = np.round(chunk)

        if self._quant_threshold is not None:
            np.clip(chunk, -self._quant_threshold, self._quant_threshold, out=chunk)

        return chunk


class TableGreyscaleDecoder(GreyscaleDecoder):
    def __init__(self, quant_table, chunk_size=8):
        assert quant_table is not None
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
    assert n > 0

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


class CompressingTableGreyscaleEncoder(TableGreyscaleEncoder):
    def __init__(self, compressor, quant_table, quant_threshold=None, chunk_size=8):
        assert compressor is not None
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


class DecompressingTableGreyscaleDecoder(TableGreyscaleDecoder):
    def __init__(self, decompressor, quant_table, chunk_size=8):
        assert decompressor is not None
        super().__init__(quant_table, chunk_size)
        self._zigzag_order = zigzag(chunk_size)
        self._decompressor = decompressor

    def apply(self, image):
        decompressed = self._decompress(image)
        super().apply(decompressed)
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


class YCbCrSubsamplingEncoder(ImageProcessor):
    def __init__(self, processors, sampling):
        assert 'Y' in processors
        assert 'Cb' in processors
        assert 'Cr' in processors

        self._processors = processors
        self._sampling = sampling

    def chroma_subsample(self, subchannel, subsampling):
        reduced = skm.block_reduce(subchannel,
                                   block_size=(subsampling[0], subsampling[1]),
                                   func=np.median)
        return np.uint8(reduced)

    def apply(self, image):
        channels = image.shape[2]

        assert 3 <= channels <= 4
        assert channels == 3 or 'A' in self._processors

        image = image
        height = image.shape[0]
        width = image.shape[1]

        # Alpha subsampling
        A = None
        if channels == 4:
            A = self.chroma_subsample(image[:, :, 3], self._sampling)
            A = self._processors['A'].apply(A)

        ycbcr = rgb2ycbcr(image[:, :, :3])

        Y = ycbcr[:, :, 0]
        Y = self._processors['Y'].apply(Y)

        Cb = self.chroma_subsample(ycbcr[:, :, 1], self._sampling)
        Cb = self._processors['Cb'].apply(Cb)

        Cr = self.chroma_subsample(ycbcr[:, :, 2], self._sampling)
        Cr = self._processors['Cr'].apply(Cr)

        return {
            'height': height,
            'width': width,
            'sampling': self._sampling,
            'Y': Y,
            'Cb': Cb,
            'Cr': Cr,
            'A': A
        }


class YCbCrSubsamplingDecoder(ImageProcessor):
    def __init__(self, processors):
        assert 'Y' in processors
        assert 'Cb' in processors
        assert 'Cr' in processors

        self._processors = processors

    def chroma_oversample(self, subchannel, image):
        sampling = image['sampling']
        oversampled = np.zeros(shape=(image['height'], image['width']))
        for i in range(oversampled.shape[0]):
            for j in range(oversampled.shape[1]):
                oversampled[i][j] = subchannel[i // sampling[0]][j // sampling[1]]
        return oversampled

    def apply(self, image):
        assert 'height' in image
        assert 'width' in image
        assert 'sampling' in image
        assert 'Y' in image and image['Y'] is not None
        assert 'Cb' in image and image['Cb'] is not None
        assert 'Cr' in image and image['Cr'] is not None
        assert 'A' in image and (image['A'] is None or 'A' in self._processors)

        reconstructed = np.zeros(shape=(image['height'],
                                        image['width'],
                                        4))

        Y = self._processors['Y'].apply(image['Y'])
        reconstructed[:, :, 0] = Y

        Cb = self._processors['Cb'].apply(image['Cb'])
        Cb = self.chroma_oversample(Cb, image)
        reconstructed[:, :, 1] = Cb

        Cr = self._processors['Cr'].apply(image['Cr'])
        Cr = self.chroma_oversample(Cr, image)
        reconstructed[:, :, 2] = Cr

        reconstructed[:, :, :3] = ycbcr2rgb(reconstructed[:, :, :3])

        if image['A'] is not None:
            A = self._processors['A'].apply(image['A'])
            A = self.chroma_oversample(A, image)
            reconstructed[:, :, 3] = A
        else:
            reconstructed = reconstructed[:, :, :3]

        return reconstructed.astype(np.uint8)


class Pad(ImageProcessor):
    def __init__(self, multiple):
        self._multiple = multiple

    def apply(self, image):
        pad0 = 0
        if image.shape[0] % self._multiple != 0:
            pad0 = self._multiple - (image.shape[0] % self._multiple)

        pad1 = 0
        if image.shape[1] % self._multiple != 0:
            pad1 = self._multiple - (image.shape[1] % self._multiple)

        pad_width = [(0, pad0), (0, pad1)]
        while len(pad_width) < len(image.shape):
            pad_width.append((0, 0))

        old_dtype = image.dtype
        image = np.pad(image,
                       pad_width=pad_width,
                       mode='constant',
                       constant_values=0)
        assert image.dtype == old_dtype

        return image


class Pipeline(ImageProcessor):
    def __init__(self, *processors):
        assert len(processors) > 0
        self._processors = processors

    def apply(self, image):
        for processor in self._processors:
            image = processor.apply(image)
        return image


class KeepShape(ImageProcessor):
    def __init__(self, inner):
        assert inner is not None
        self._inner = inner

    def apply(self, image):
        original_shape = image.shape

        image = self._inner.apply(image)

        # Crop image
        assert len(original_shape) == len(image.shape)
        assert len(original_shape) in [2, 3]
        assert all([original_shape[i] <= image.shape[i] for i in range(len(original_shape))])

        if len(original_shape) == 2:
            image = image[:original_shape[0], :original_shape[1]]
        else:
            image = image[:original_shape[0], :original_shape[1], :original_shape[2]]

        return image


def rgb2ycbcr(rgb):
    """
    In accordance with http://www.itu.int/rec/T-REC-T.871-201105-I/en
    """
    conversion_matrix = np.array([
        [.299, .587, .114],
        [-.168736, -.331264, .5],
        [.5, -.418688, -.081312]])
    rgb = rgb.astype(np.double)
    rgb = rgb.dot(conversion_matrix.T)
    rgb[:, :, [1, 2]] += 128
    rgb = np.round(rgb)
    rgb = np.clip(rgb, 0, 255)
    return np.uint8(rgb)


def ycbcr2rgb(ycbcr):
    """
    In accordance with http://www.itu.int/rec/T-REC-T.871-201105-I/en
    """
    conversion_matrix = np.array([
        [1, 0, 1.402],
        [1, -0.344136, -.714136],
        [1, 1.772, 0]])
    ycbcr = ycbcr.astype(np.double)
    ycbcr[:, :, [1, 2]] -= 128
    ycbcr = ycbcr.dot(conversion_matrix.T)
    ycbcr = np.round(ycbcr)
    ycbcr = np.clip(ycbcr, 0, 255)
    return np.uint8(ycbcr)


def psnr(signal, noise):
    assert signal.shape == noise.shape
    max_I = 255
    mse = np.mean(np.subtract(signal, noise, dtype=np.double) ** 2, axis=None)
    return 20 * np.log10(max_I) - 10 * np.log10(mse)


def build_quant_table(args):
    # TODO: write proper
    quant_table = np.array([[17, 18, 24, 47, 99, 128, 192, 256],
                            [18, 21, 26, 66, 99, 192, 256, 512],
                            [24, 26, 56, 99, 128, 256, 512, 512],
                            [47, 66, 99, 128, 256, 512, 1024, 1024],
                            [99, 99, 128, 256, 512, 1024, 2048, 2048],
                            [128, 192, 256, 512, 1024, 2048, 4096, 4096],
                            [192, 256, 512, 1024, 2048, 4096, 8192, 8192],
                            [256, 512, 512, 1024, 2048, 4096, 8192, 8192]])

    quant_table = np.array([[1, 1, 1, 16, 24, 40, 51, 61],
                            [1, 1, 1, 19, 26, 58, 60, 55],
                            [1, 1, 1, 24, 40, 57, 69, 56],
                            [14, 17, 22, 29, 51, 87, 80, 62],
                            [18, 22, 37, 56, 68, 109, 103, 77],
                            [24, 35, 55, 64, 81, 104, 113, 92],
                            [49, 64, 78, 87, 103, 121, 120, 101],
                            [72, 92, 95, 98, 112, 100, 103, 99]])

    quant_table = 1 * np.ones_like(quant_table)

    return quant_table


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Compress and decompress images using JPEG techniques.')

    # Modes of operation
    parser.add_argument('-c', '--compress', action='store_true', help='Compress')
    parser.add_argument('-d', '--decompress', action='store_true', help='Decompress')
    parser.add_argument('-p', '--psnr', action='store_true', help='Peak signal to noise ratio between original and compressed images')
    parser.add_argument('-r', '--rate', action='store_true', help='Compression rate between original and compressed images')

    # Parameters
    parser.add_argument('-b', '--block-size', default=8, help='Which block size to use for compression', type=int)
    parser.add_argument('-q', '--quant-coefficient', default=50, help='Quantization factor to use', type=float)
    parser.add_argument('-u', '--quant-threshold', default=2000, help='Quantization threshold to use', type=float)
    parser.add_argument('-t', '--table', help='Quantization table preset to use when not using a quantization coefficient', type=str, choices=['default', 'all', 'ac_only'])

    parser.add_argument('--huffman', action='store_false', help='Do not use Huffman when compressing')
    parser.add_argument('--huffman-tree', default='default', help='Huffman tree preset to use when compressing', type=str, choices=['default'])

    parser.add_argument('input', type=str, nargs=1, help='Input file to compress')
    parser.add_argument('output', type=str, nargs='?', help='File to dump output')
    args = parser.parse_args()

    args.input = args.input[0]
    if args.psnr:
        assert args.output is not None
        left = ski.io.imread(args.input)
        right = ski.io.imread(args.output)
        print(psnr(left, right))
        exit(0)
    elif args.rate:
        original_image = ski.io.imread(args.input)
        rate = float(original_image.shape[0] * original_image.shape[1]) / os.stat(args.output).st_size
        print(rate)
        exit(0)
    elif not (args.compress or args.decompress):
        raise NotImplementedError('Unknown operation mode')

    # Prepares the codecs
    chunk_size = args.block_size
    quant_threshold = args.quant_threshold
    quant_table = None
    if args.table is not None:
        quant_table = build_quant_table(args)
    else:
        quant_table = args.quant_coefficient * np.ones(shape=(chunk_size, chunk_size))

    compressor = None
    if args.huffman_tree is not None:
        # TODO: esto tendria que ser un buen compresor, hay que usar una tabla copada. Esta tabla flashea fuerte.
        compressor = HuffmanTree.from_weight_dictionary({4: 54})

    greyscale_encoder = None
    greyscale_decoder = None
    # By default, we are compressing a greyscale image
    if args.huffman:
        assert compressor is not None
        greyscale_encoder = CompressingTableGreyscaleEncoder(compressor, quant_table, quant_threshold, chunk_size)
        greyscale_decoder = DecompressingTableGreyscaleDecoder(compressor, quant_table, chunk_size)
    else:
        greyscale_encoder = TableGreyscaleEncoder(quant_table, quant_threshold, chunk_size)
        greyscale_decoder = TableGreyscaleDecoder(quant_table, chunk_size)

    # Handles non divisible by chunk_size images coming into the greyscale encoder; to prevent chunking from failing.
    greyscale_encoder = Pipeline(Pad(chunk_size), greyscale_encoder)

    # We are compressing an image in RGB or RGBA
    color_encoder = YCbCrSubsamplingEncoder({
        'Y': greyscale_encoder,
        'Cb': greyscale_encoder,
        'Cr': greyscale_encoder,
        'A': greyscale_encoder}, (2, 2)) # TODO: add sampling as a parameter

    # Handles non divisible by chunk_size images coming into the YCbCr encoder. Prevents chunking from failing. Notice
    # that we have another Pad down the line in each encoder, that is because after subsampling, the images may not be
    # divisible by chunk size, so we need to handle that case.
    color_encoder = Pipeline(Pad(chunk_size), color_encoder)

    color_decoder = YCbCrSubsamplingDecoder({
        'Y': greyscale_decoder,
        'Cb': greyscale_decoder,
        'Cr': greyscale_decoder,
        'A': greyscale_decoder
    })

    if args.compress:
        # In this case, we just produced a file in our own format containing relevant information for decoding it
        raw_image = ski.io.imread(args.input)

        compressed_image = None
        if len(raw_image.shape) == 2:
            compressed_image = greyscale_encoder.apply(raw_image)
        else:
            assert len(raw_image.shape) == 3
            assert raw_image.shape[2] in [3, 4]
            compressed_image = color_encoder.apply(raw_image)

        with open(args.output, 'wb') as handle:
            pickle.dump({
                'kind': len(raw_image.shape),
                'image': compressed_image
            }, handle)
    elif args.decompress:
        compressed_image = None
        with open(args.input, 'rb') as handle:
            compressed_image = pickle.load(handle)

        kind = compressed_image['kind']
        assert kind in [2, 3]
        compressed_image = compressed_image['image']

        if kind == 3:
            raw_image = color_decoder.apply(compressed_image)
        else:
            raw_image = greyscale_decoder.apply(compressed_image)
        ski.io.imsave(args.output, raw_image)
    else:
        raise NotImplementedError('Unknown operation mode')
