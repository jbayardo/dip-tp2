#!/usr/bin/env python2
import sys
import heapq
import struct

import bitarray

import PIL
import PIL.Image

import numpy
import scipy.fftpack

def img(fn):
    return PIL.Image.open(fn).convert('L')

def wrap(x):
    return min(255, max(0, int(x)))

def pad_image(img, pwidth, pheight):
    """
    Pad an image so its width is a multiple of pwidth and
    its height is a multiple of pheight.
    """
    new_width = (img.width // pwidth) * pwidth
    if img.width % pwidth > 0:
        new_width += pwidth
    new_height = (img.height // pheight) * pheight
    if img.height % pheight > 0:
        new_height += pheight
    img_padded = PIL.Image.new('L', (new_width, new_height))
    img_padded.paste(img)
    return img_padded

def inverse_pad_image(img_size, img_padded):
    "Crop a padded image to recover the original image."
    return img_padded.crop([0, 0, img_size[0], img_size[0]])

def image_to_matrix(img):
    "Return a numpy matrix such that matrix[i][j] == img.getpixel((i,j))"
    matrix = numpy.fromstring(img.tobytes(), dtype=numpy.uint8)
    return matrix.reshape((img.height, img.width)).T

def inverse_image_to_matrix(matrix):
    "Return an image such that matrix[i][j] == img.getpixel((i,j))"
    width = len(matrix)
    height = len(matrix[0])
    img = PIL.Image.new('L', (width, height))
    matrix = matrix.T.reshape(width * height)
    matrix = numpy.array([numpy.uint8(x) for x in matrix])
    img.frombytes(matrix.tostring())
    return img

def split_blocks(img, n):
    """
    Split an image in tiles of size n * m, returning a matrix of blocks.
    The width (resp. height) of the image is supposed to be a multiple of
    n (resp. m).
    """
    def get_block(img, i, j, n):
        img2 = PIL.Image.new('L', (n, n))
        img2.paste(img.crop([i, j, i + n, j + n]))
        return img2
    block_matrix = []
    for i in range(img.width / n):
        block_row = []
        for j in range(img.height / n):
            block_row.append(get_block(img, n * i, n * j, n))
        block_matrix.append(block_row)
    return block_matrix

def inverse_split_blocks(block_matrix):
    """
    Reconstruct an image from the matrix of blocks.
    """
    width = len(block_matrix) * block_matrix[0][0].width
    height = len(block_matrix[0]) * block_matrix[0][0].height
    img = PIL.Image.new('L', (width, height))
    for i in range(len(block_matrix)):
        for j in range(len(block_matrix[0])):
            block = block_matrix[i][j]
            img.paste(block, (block.width * i, block.height * j))
    return img

def dct2(matrix):
    "Perform the 2-dimensional discrete cosine transform."
    return scipy.fftpack.dct(
             scipy.fftpack.dct(matrix.T, norm='ortho').T,
             norm='ortho'
           )

def inverse_dct2(matrix):
    "Perform the inverse 2-dimensional discrete cosine transform."
    return scipy.fftpack.idct(
             scipy.fftpack.idct(matrix.T, norm='ortho').T,
             norm='ortho'
           )

def blocks_dct2(block_matrix):
    "Perform the DCT on each block of the matrix."
    dct_matrix = []
    for block_row in block_matrix:
        dct_row = []
        for block in block_row:
            dct_row.append(dct2(image_to_matrix(block)))
        dct_matrix.append(numpy.array(dct_row))
    return numpy.array(dct_matrix)

def inverse_blocks_dct2(dct_matrix):
    "Perform the inverse DCT on each block of the matrix."
    block_matrix = []
    for dct_row in dct_matrix:
        block_row = []
        for dct_elem in dct_row:
            block_row.append(inverse_image_to_matrix(inverse_dct2(dct_elem)))
        block_matrix.append(block_row)
    return block_matrix

def map_blocks(f, matrix):
    "Apply a function on each element of the matrix."
    result_matrix = []
    for row in matrix:
        result_row = []
        for elem in row:
            result_row.append(f(elem))
        if type(row) != 'list':
            result_row = numpy.array(result_row)
        result_matrix.append(result_row)
    if type(matrix) != 'list':
        result_matrix = numpy.array(result_matrix)
    return result_matrix

def quantize_blocks(quantization_factor, quantization_threshold, dct_matrix):
    """
    Apply quantization on each block of the matrix,
    and.
    This is the only lossy component in the compression pipeline.
    """
    def quantize_block(block):
        qblock = [] 
        for row in block:
            qrow = []
            for elem in row:
                qelem = int(elem / quantization_factor)
                if qelem > quantization_threshold:
                    qelem = quantization_threshold
                elif qelem < -quantization_threshold:
                    qelem = -quantization_threshold
                qrow.append(qelem)
            qblock.append(qrow)
        return qblock
    return map_blocks(quantize_block, dct_matrix)

def inverse_quantize_blocks(quantization_factor, dct_matrix_quantized):
    "Apply inverse quantization on each block of the matrix."
    def inverse_quantize_block(qblock):
        block = [] 
        for qrow in qblock:
            row = []
            for qelem in qrow:
                elem = qelem * quantization_factor
                row.append(elem)
            block.append(row)
        return block
    return map_blocks(inverse_quantize_block, dct_matrix_quantized)

def zigzag_indices(n, m):
    """
    Return a permutation of the list of all indices (i, j) with
    0 <= i < n and 0 <= j < n, according to the diagonal walk.
    """
    i, j = 0, 0 
    yield i, j
    while (i, j) != (n - 1, m - 1):
        if j < m - 1:
            j += 1
        else:
            i += 1
        yield i, j
        while j > 0 and i < n - 1:
            i += 1
            j -= 1
            yield i, j
        if i < n - 1:
            i += 1
        else:
            j += 1
        yield i, j
        while i > 0 and j < m - 1:
            i -= 1
            j += 1
            yield i, j

def dc_coding(dct_matrix_quantized):
    "Encode the matrix of blocks using DC coding representation."
    def dc_coding_block(previous_dc, block):
        width = len(block)
        height = len(block[0])
        block_dc = []
        for i, j in zigzag_indices(width, height):
            value = block[i][j]
            if (i, j) == (0, 0):
                value -= previous_dc
            block_dc.append(value)
        return block_dc
    dct_matrix_dc = []
    previous_dc = 0
    for ii in range(len(dct_matrix_quantized)):
        for jj in range(len(dct_matrix_quantized)):
            dct_matrix_dc.extend(
              dc_coding_block(previous_dc, dct_matrix_quantized[ii][jj])
            )
            previous_dc = dct_matrix_quantized[ii][jj][0][0]
    return dct_matrix_dc

def inverse_dc_coding(img_size, block_size, dct_matrix_dc):
    "Decode a matrix from the DC coding representation."
    img_width, img_height = img_size 

    def next_inverse_dc_coding_block(previous_dc, curr_index):
        n = 0
        block = numpy.zeros((block_size, block_size))
        for i, j in zigzag_indices(block_size, block_size):
            value = dct_matrix_dc[curr_index]
            if (i, j) == (0, 0):
                value += previous_dc
            block[i][j] = value
            curr_index += 1
        return block, curr_index

    curr_index = 0
    dct_matrix_quantized = []
    previous_dc = 0
    for ii in range(img_width // block_size):
        row = []
        for jj in range(img_height // block_size):
            block, curr_index = next_inverse_dc_coding_block(
                                  previous_dc, curr_index
                                )
            row.append(block)
            previous_dc = block[0][0]
        dct_matrix_quantized.append(row)
    return dct_matrix_quantized

def huffman(dct_matrix_dc):
    "Compress a list of numbers using Huffman coding."
    # Count frequencies
    d = {}
    for v in dct_matrix_dc:
        d[v] = d.get(v, 0) + 1

    # Build Huffman tree
    q = [(v, k) for (k, v) in d.items()]
    heapq.heapify(q)
    while len(q) > 1:
        v1, k1 = heapq.heappop(q)
        v2, k2 = heapq.heappop(q)
        heapq.heappush(q, [v1 + v2, [k1, k2]])
    huff_tree = q[0][1]

    # Build Huffman table
    def build_huffman_table(tree, table, prefix):
        if isinstance(tree, list):
            table = build_huffman_table(tree[0], table, prefix + [0])
            table = build_huffman_table(tree[1], table, prefix + [1])
            return table
        else:
            table[tree] = prefix
            return table
    huff_table = build_huffman_table(huff_tree, {}, [])

    # Build bitarray
    huff_bits = bitarray.bitarray()
    for value in dct_matrix_dc:
        huff_bits.extend(huff_table[value])

    return huff_tree, huff_bits

def inverse_huffman(huff_tree, huff_bits):
    "Decompress a Huffman-encoded list of numbers."
    dct_matrix_dc = []
    p = huff_tree 
    for bit in huff_bits:
        p = p[1] if bit else p[0]
        if not isinstance(p, list):
            dct_matrix_dc.append(p)
            p = huff_tree
    return dct_matrix_dc

def pack_data(img_size, img_padded_size, block_size, quantization_factor,
              huff_tree, huff_bits):
    "Pack all the data representing the compressed image into a string."
    def write_huff_tree(tree):
        if isinstance(tree, list):
            return ':' + write_huff_tree(tree[0]) \
                       + write_huff_tree(tree[1])
        else:
            return '.' + struct.pack('i', tree)
    compressed_image = []
    compressed_image.append(struct.pack('i', img_size[0]))
    compressed_image.append(struct.pack('i', img_size[1]))
    compressed_image.append(struct.pack('i', img_padded_size[0]))
    compressed_image.append(struct.pack('i', img_padded_size[1]))
    compressed_image.append(struct.pack('i', block_size))
    compressed_image.append(struct.pack('i', quantization_factor))
    compressed_image.append(write_huff_tree(huff_tree))
    compressed_image.append(struct.pack('i', len(huff_bits)))
    compressed_image.append(huff_bits.tobytes())
    return ''.join(compressed_image)

def inverse_pack_data(compressed_image):
    "Unpack all the data representing the compressed image from a string."
    def parse_huff_tree(string, i):
        if string[i] == ':':
            i += 1
            i, left = parse_huff_tree(string, i)
            i, right = parse_huff_tree(string, i)
            return i, [left, right]
        elif string[i] == '.':
            i += 1
            value = struct.unpack('i', string[i : i + 4])[0]
            i += 4
            return i, value
        else:
            raise Exception()
    i = 0
    (
      img_width,
      img_height,
      img_padded_width,
      img_padded_height,
      block_size,
      quantization_factor,
    ) = struct.unpack('iiiiii', compressed_image[i : i + 24])
    i += 24
    i, huff_tree = parse_huff_tree(compressed_image, i)
    (nbits,) = struct.unpack('i', compressed_image[i : i + 4])
    i += 4
    huff_bits = bitarray.bitarray()
    huff_bits.frombytes(compressed_image[i : ])
    huff_bits = huff_bits[: nbits]
    return (
        (img_width, img_height),
        (img_padded_width, img_padded_height),
        block_size,
        quantization_factor,
        huff_tree,
        huff_bits
    )

def simple_jpeg_compression(img,
                            block_size=8,
                            quantization_factor=10,
                            quantization_threshold=1000,
                           ):
    "Compress an image using the given parameters."
    img_padded = pad_image(img, block_size, block_size)
    block_matrix = split_blocks(img_padded, block_size)
    dct_matrix = blocks_dct2(block_matrix)
    dct_matrix_quantized = quantize_blocks(
                             quantization_factor,
                             quantization_threshold,
                             dct_matrix
                           )
    dct_matrix_dc = dc_coding(dct_matrix_quantized)
    huff_tree, huff_bits = huffman(dct_matrix_dc)
    compressed_image = pack_data(
                         img.size,
                         img_padded.size,
                         block_size,
                         quantization_factor,
                         huff_tree,
                         huff_bits
                       )
    return compressed_image

def simple_jpeg_decompression(data):
    "Decompress an image."
    compressed_image = data
    (
      img_size,
      img_padded_size,
      block_size,
      quantization_factor,
      huff_tree,
      huff_bits
    ) = inverse_pack_data(compressed_image)
    dct_matrix_dc = inverse_huffman(huff_tree, huff_bits)
    dct_matrix_quantized = inverse_dc_coding(
                             img_padded_size,
                             block_size,
                             dct_matrix_dc
                           )
    dct_matrix = inverse_quantize_blocks(
                   quantization_factor,
                   dct_matrix_quantized
                 )
    block_matrix = inverse_blocks_dct2(dct_matrix)
    img_padded = inverse_split_blocks(block_matrix)
    img = inverse_pad_image(img_size, img_padded)
    return img

def usage():
    sys.stderr.write('Usage:\n')
    sys.stderr.write('  %s -c in.png out.j    Compress.\n' % (sys.argv[0]))
    sys.stderr.write('  %s -d in.j out.png    Decompress.\n' % (sys.argv[0]))
    sys.stderr.write('Options for compression:\n')
    sys.stderr.write('  -b block_size[=8]\n')
    sys.stderr.write('  -f quantization_factor[=50]\n')
    sys.stderr.write('  -t quantization_threshold[=2000]\n')

def main():
    argv = sys.argv[:]
    argv.pop(0)

    args = []
    options = {
        'block_size': 8,
        'quantization_factor': 50,
        'quantization_threshold': 2000,
    }
    while len(argv) > 0:
        opt = argv.pop(0)
        if opt == '-b':
            if len(argv) == 0: usage()
            options['block_size'] = int(argv.pop(0))
        elif opt == '-f':
            if len(argv) == 0: usage()
            options['quantization_factor'] = int(argv.pop(0))
        elif opt == '-t':
            if len(argv) == 0: usage()
            options['quantization_threshold'] = int(argv.pop(0))
        else:
            args.append(opt)

    if len(args) == 3 and args[0] == '-c':
        infile, outfile = args[1:]
        with open(outfile, 'w') as f:
            f.write(
                simple_jpeg_compression(
                  img(infile),
                  block_size=options['block_size'],
                  quantization_factor=options['quantization_factor'],
                  quantization_threshold=options['quantization_threshold'],
                )
            )
    elif len(args) == 3 and args[0] == '-d':
        infile, outfile = args[1:]
        with open(infile, 'r') as f:
            simple_jpeg_decompression(f.read()).save(outfile)
    else:
        usage()

if __name__ == '__main__':
    main()

