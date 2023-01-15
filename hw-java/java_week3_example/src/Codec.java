/**
 * Provided methods for encode and decode of data elements for HW1.
 */

class Codec {
    /**
     * Provided encode function Just does something time-consuming (and arbitrary)
     *
     * @param v number to encode
     * @return the encoded number
     */
    static int encode(int v) {
        for (int i = 0; i < 500; i++)
            v = ((v * v) + v) % 10;
        return v;
    }

    /**
     * Provided decode function Just does something time-consuming (and arbitrary)
     *
     * @param v encoded number to decode
     * @return the decoded number
     */
    static int decode(int v) {
        return encode(v);
    }
}

