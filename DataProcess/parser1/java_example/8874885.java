public class Test8874885 {
    public HashSet<Character> getKanji(String s, String encoding) throws Exception {
        HashSet<Character> hs = new HashSet<Character>();
        if (encoding == null) encoding = "UTF-8";
        if (!s.startsWith("http")) {
            throw new Exception("Url must point to a text or html file, and must use http protocol");
        }
        try {
            String result = null;
            URL url = new URL(s);
            URLConnection connection = url.openConnection();
            connection.setRequestProperty("User-Agent", "Mozilla/4.0 (compatible; MSIE 6.0; Windows NT 5.1)");
            connection.setDoOutput(false);
            BufferedReader in = new BufferedReader(new InputStreamReader(connection.getInputStream(), encoding));
            String inputLine;
            String contentType = connection.getContentType();
            if (!(contentType.startsWith("text") || contentType.startsWith("application/xml"))) {
                in.close();
                throw new Exception("Url must point to a text or html file, and must use http protocol");
            }
            while ((inputLine = in.readLine()) != null) {
                char[] arr = inputLine.toCharArray();
                for (int i = 0; i < arr.length; i++) {
                    if (arr[i] >= '々' && arr[i] <= '〇' || arr[i] >= '' && arr[i] <= '') {
                        if (!hs.contains(arr[i])) {
                            hs.add(arr[i]);
                        }
                    }
                }
            }
            in.close();
        } catch (Exception e) {
            e.printStackTrace();
            throw e;
        }
        return hs;
    }

}