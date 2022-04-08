
const image = document.getElementById("image");

;(async () => {
  await tf.setBackend("wasm")
  const model = await tf.loadGraphModel("/converted/model.json");
  const start  = performance.now()
  for(let i = 0; i < 100; i++) {
    const predictions = await model.predict(
      tf.div(tf.cast(tf.browser.fromPixels(image, 1), "float32"), 255).expandDims()
    );
    const valArr = await predictions.argMax([-1]).data()
    const result = valArr[0]
    document.getElementById("pre").innerText = result
  }
  console.log((performance.now() - start) / 100, "ms / detection")
})()