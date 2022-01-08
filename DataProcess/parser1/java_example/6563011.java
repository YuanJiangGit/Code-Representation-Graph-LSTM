public class Test6563011 {
    public void doApacheHttpRequest(MethodOfProxySelection proxySelect, MethodOfProxyAuthentication proxyAuthenticate) throws ConfProblem {
        log("-------------------- Request ------------------------------");
        log("proxy selection method:        " + proxySelect);
        log("proxy authentication method:   " + proxyAuthenticate);
        DefaultHttpClient httpclient = new DefaultHttpClient();
        switch(proxySelect) {
            case manual:
                HttpHost proxyHost = null;
                try {
                    proxyHost = conf.constructManualProxy();
                } catch (ConfProblem p) {
                    p.print();
                    return;
                }
                httpclient.getParams().setParameter(ConnRoutePNames.DEFAULT_PROXY, proxyHost);
                break;
            case detect_proxyvole:
                ProxySearch proxySearch = ProxySearch.getDefaultProxySearch();
                ProxySelector myProxySelector = proxySearch.getProxySelector();
                ProxySelector.setDefault(myProxySelector);
            case detect_builtin:
                ProxySelectorRoutePlanner routePlanner = new ProxySelectorRoutePlanner(httpclient.getConnectionManager().getSchemeRegistry(), ProxySelector.getDefault());
                httpclient.setRoutePlanner(routePlanner);
                List<Proxy> proxies = ProxySelector.getDefault().select(message.uri);
                log(proxies.size() + " discovered proxies:- ");
                for (Proxy p : proxies) {
                    log("    " + p);
                }
                break;
            default:
        }
        switch(proxyAuthenticate) {
            case ntlm:
                {
                    NTCredentials creds = conf.constructNtlmCredentials();
                    httpclient.getCredentialsProvider().setCredentials(AuthScope.ANY, creds);
                    break;
                }
            case basic:
                {
                    UsernamePasswordCredentials credentials = conf.constructBasicCredentials();
                    httpclient.getCredentialsProvider().setCredentials(AuthScope.ANY, credentials);
                    break;
                }
            case none:
            default:
        }
        HttpParams params = new BasicHttpParams();
        HttpProtocolParams.setVersion(params, HttpVersion.HTTP_1_1);
        HttpProtocolParams.setContentCharset(params, "UTF-8");
        HttpProtocolParams.setUserAgent(params, "Vexi");
        HttpProtocolParams.setUseExpectContinue(params, true);
        BasicHttpEntityEnclosingRequest request = new BasicHttpEntityEnclosingRequest("POST", message.uriString);
        ByteArrayEntity entity = new ByteArrayEntity(message.message.getBytes());
        entity.setContentType("text/xml");
        request.setEntity(entity);
        HttpHost host = new HttpHost(message.host, message.port);
        request.setParams(params);
        try {
            log("-------------------- Response ------------------------------");
            HttpResponse response = httpclient.execute(host, request);
            byte[] bytes = IOUtil.toByteArray(response.getEntity().getContent());
            int statusCode = response.getStatusLine().getStatusCode();
            log("Status Code: " + statusCode);
            log(new String(bytes));
        } catch (IOException e) {
            System.err.println("[Problem] Exception thrown performing request");
            e.printStackTrace();
        }
    }

}