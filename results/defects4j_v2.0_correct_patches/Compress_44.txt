public ChecksumCalculatingInputStream(final Checksum checksum, final InputStream in) {
    if (checksum == null) {
        throw new NullPointerException("checksum");
    }
    if (in == null) {
        throw new NullPointerException("in");
    }
    this.checksum = checksum;
    this.in = in;
}