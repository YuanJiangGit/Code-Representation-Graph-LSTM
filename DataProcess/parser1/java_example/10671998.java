public class Test10671998 {
    public static BufferedImage ImageIORead(File file) {
        System.gc();
        try {
            FileChannel roChannel = new RandomAccessFile(file, "r").getChannel();
            final ByteBuffer buf = roChannel.map(FileChannel.MapMode.READ_ONLY, 0, (int) roChannel.size());
            return ImageIO.read(new InputStream() {

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
            });
        } catch (Exception ex) {
            Tools.logException(Tools.class, ex, file.getAbsolutePath());
        }
        try {
            return ImageIO.read(new FileInputStream(file));
        } catch (Exception ex) {
            Tools.logException(Tools.class, ex, file.getAbsolutePath());
        }
        return null;
    }

}