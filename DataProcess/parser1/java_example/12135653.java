public class Test12135653 {
    public static URL createFacelectUrl(String viewId) {
        try {
            return new URL(null, PROTOCOL + viewId, new URLStreamHandler() {

                @Override
                protected URLConnection openConnection(URL url) throws IOException {
                    return new URLConnection(url) {

                        @Override
                        public void connect() throws IOException {
                        }

                        @Override
                        public InputStream getInputStream() throws IOException {
                            ThemeResource f = getFacelet(getURL().getFile());
                            return new ByteArrayInputStream(f.getText().getBytes());
                        }
                    };
                }
            });
        } catch (MalformedURLException e) {
            throw new RuntimeException(e);
        }
    }

}