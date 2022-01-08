public class Test10671995 {
    public static InputStream getInputStream(File file) {
        try {
            FileChannel roChannel = new RandomAccessFile(file, "r").getChannel();
            final ByteBuffer buf = roChannel.map(FileChannel.MapMode.READ_ONLY, 0, (int) roChannel.size());
            return new InputStream() {

                public synchronized int read() throws IOException {
                    if (!buf.hasRemaining()) {
                        return -1;
                    }
                    return buf.get();
                }

                public synchronized int read(byte[] bytes, int off, int len) throws IOException {
                    if (!buf.hasRemaining()) {
                        return -1;
                    }
                    len = Math.min(len, buf.remaining());
                    buf.get(bytes, off, len);
                    return len;
                }
            };
        } catch (Exception ex) {
            Tools.logException(Tools.class, ex, file.getAbsolutePath());
        }
        return null;
    }

}