const sayHello = async () => {
    const response = await fetch('/hello', {
      method: 'POST',
      body: '{ "name": "Alice" }',
      headers: {
        'Content-Type': 'application/json'
      }
    });

    // extract JSON from the http response
    const resp = await response.json();
    return resp;
}
