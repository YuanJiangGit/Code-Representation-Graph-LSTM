public class Test11896536 {
    private byte[] invokeServiceAndGsetXMLByteArrayFromInput(DPWSContextImpl context, BindingProvider provider, MOREService serviceInstance, String service, String operation, List<Object> params) throws Exception {
        ByteBuffer allocated = null;
        try {
            class invocationThread extends Thread {

                public boolean success;

                DPWSContextImpl context;

                BindingProvider provider;

                MOREService serviceInstance;

                String service;

                String operation;

                List<Object> params;

                PipedOutputStream outStream;

                public invocationThread(PipedOutputStream outStream, DPWSContextImpl context, BindingProvider provider, MOREService serviceInstance, String service, String operation, List<Object> params) {
                    success = true;
                    this.context = context;
                    this.provider = provider;
                    this.serviceInstance = serviceInstance;
                    this.service = service;
                    this.operation = operation;
                    this.params = params;
                    this.outStream = outStream;
                }

                public void run() {
                    try {
                        Object[] invokeReturnObject = serviceInstance.sendToOtherServiceOnNode(context, service, operation, params);
                        if ((invokeReturnObject != null) && (invokeReturnObject[0] != null)) {
                            final XMLStreamWriter writer = STAXUtils.createXMLStreamWriter(outStream, null, context);
                            provider.writeParameter(null, writer, context, invokeReturnObject[0]);
                            writer.close();
                        } else throw new Exception("Parsing not successful");
                    } catch (Exception e) {
                        System.out.println("Thread-Shit: " + e);
                        success = false;
                    }
                }
            }
            PipedInputStream stream = new PipedInputStream();
            invocationThread writeThread = new invocationThread(new PipedOutputStream(stream), context, provider, serviceInstance, service, operation, params);
            writeThread.start();
            byte[] buffer = new byte[4096];
            while (true) {
                if ((stream.available() > 0) && (allocated == null)) {
                    int laenge = stream.read(buffer);
                    allocated = (ByteBuffer.allocate(laenge)).put(buffer, 0, laenge);
                }
                if ((allocated != null) || (!writeThread.success)) break;
            }
        } catch (Exception e) {
        }
        if (allocated != null) return allocated.array(); else throw new Exception();
    }

}