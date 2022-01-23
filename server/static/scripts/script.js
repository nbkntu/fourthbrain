const getBoundingBoxes = async (filename) => {
    const response = await fetch('/get_bounding_boxes', {
      method: 'POST',
      body: `{
        "image_id": "img1",
        "image_file_name": "${filename}"
      }`,
      headers: {
        'Content-Type': 'application/json'
      }
    });

    // extract JSON from the http response
    const resp = await response.json();
    return resp;
};
