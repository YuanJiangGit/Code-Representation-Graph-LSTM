public class Test11644243 {
    public static final <D extends BatchedData<D>> Promise<Long> forward(final AInput<D> in, final AOutput<D> out, final long length, final int readDataSize, final boolean autoflush) {
        if (in == null) {
            throw new NullPointerException("Input must not be null.");
        }
        if (out == null) {
            throw new NullPointerException("Output must not be null.");
        }
        if (readDataSize < 1) {
            throw new IllegalArgumentException("readDataSize must be positive: " + readDataSize);
        }
        if (length < -1) {
            throw new IllegalArgumentException("Length must be UNLIMITED or non-negative: " + length);
        }
        return new AsyncProcess<Long>() {

            long remainingLength = length;

            long copiedLength = 0;

            @Override
            public void run() {
                if (remainingLength == 0) {
                    success(copiedLength);
                    return;
                }
                run(null);
            }

            /**
			 * The read-write cycle. Note that cycle does read and write
			 * operation with data from previous cycle operations at the same
			 * time.
			 * 
			 * @param data
			 *            null or data to write
			 */
            void run(D data) {
                final int toRead;
                if (remainingLength != UNLIMITED && readDataSize > remainingLength) {
                    toRead = (int) remainingLength;
                } else {
                    toRead = readDataSize;
                }
                final Promise<Object[]> readAndWrite = Wait.all(readData(in, toRead), writeData(out, data));
                new ProcessWhen<Object[]>(readAndWrite) {

                    @Override
                    protected Promise<Void> resolved(Object[] value) throws Throwable {
                        D data = IOUtils.<D>blindCast(value[0]);
                        if (data == null) {
                            success(copiedLength);
                        } else {
                            copiedLength += data.length();
                            if (remainingLength != UNLIMITED) {
                                remainingLength -= data.length();
                            }
                            run(data);
                        }
                        return null;
                    }
                };
            }

            /**
			 * Read data if there is still something to read
			 * 
			 * @param in
			 *            an input stream
			 * @param toRead
			 *            amount to read
			 * @return promise for input stream or promise for null if limit is
			 *         reached.
			 */
            private Promise<D> readData(final AInput<D> in, final int toRead) {
                if (toRead == 0) {
                    return Promise.nullPromise();
                } else {
                    return in.read(toRead);
                }
            }

            /**
			 * Write data and flush it if autoflush is true
			 * 
			 * @param out
			 *            output stream
			 * @param data
			 *            a data to write
			 * @return a promise that resolves when data is written
			 */
            private Promise<Void> writeData(final AOutput<D> out, D data) {
                if (data == null) {
                    return null;
                }
                if (autoflush) {
                    return new When<Void, Void>(out.write(data)) {

                        protected Promise<Void> resolved(Void value) throws Throwable {
                            return out.flush();
                        }
                    }.promise();
                } else {
                    return out.write(data);
                }
            }
        }.promise();
    }

}