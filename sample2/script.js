
    console.log('トレーニングの第1ステップ：曲線を合成データにフィットさせる');
    // トレーニングの第1ステップ：曲線を合成データにフィットさせる
    // https://github.com/tensorflow/tfjs-examples

    // Y = a × X ^ 3 + b × X ^ 2 + c × X + d
    // という3次の方程式を考える。
    // 係数 a, b, c, と定数dが不定だということでとりあえずランダム値を与えておく。
    const a = tf.variable(tf.scalar(Math.random()));
    const b = tf.variable(tf.scalar(Math.random()));
    const c = tf.variable(tf.scalar(Math.random()));
    const d = tf.variable(tf.scalar(Math.random()));

    console.log('a-dの値（初期値）');
    // print
    a.print();
    b.print();
    c.print();
    d.print();
    // 何かの値が4つ出力されることがわかる。

    // オプティマイザは学習レートは0.5として、tf.train.sgdから生成する。
    // どうやら他にもいくつかのトレーニングオブジェクトが存在するらしいが、ここではSGDだけで済ます。
    const learningRate = 0.5;
    const optimizer = tf.train.sgd(learningRate);

    // モデルを定義する
    function predict(x) {
      // y = a * x ^ 3 + b * x ^ 2 + c * x + d
      // 上記の計算式をTFの関係式に落とし込むと次のようになる。
      // a,b,c,dは事前に決定した変数を当てている。
      return tf.tidy(() => {
        return a.mul(x.pow(tf.scalar(3, 'int32')))
          .add(b.mul(x.square()))
          .add(c.mul(x))
          .add(d);
      });
    }
    // モデルの動作を確認してみる。
    // 例として値に2を与えたら
    console.log('損失関数の出力の例（Input:2）');
    const r0 = predict(tf.scalar(2));
    r0.print();  // 与えたXにて計算されたYの値が出力されることがわかる。

    // さて、次に損失関数を定義しましょう。
    // この関数は実際の値とモデルで予測した値の差をエラー値（誤差）として戻すだけのものです。
    // よく最小二乗法とかで利用される「差の二乗の平均」を取るには
    //   (予測値 - 実測値) ^ 2 の全体平均
    // を prediction.sub(labels).square().mean() で計算する。
    function loss(prediction, labels) {
      const error = prediction.sub(labels).square().mean();
      return error;
    }

    // ここで学習ロジックを作る。
    // 100回のループで、オプティマイザのminimizeメソッドを呼び出して、損失関数の戻りを戻している。
    // xs,ysは不変なので、ここではa,b,c,dの値が変更されているということになるのだろう。
    const numIterations = 100;
    function train(xs, ys, numIterations) {
      for (let iter = 0; iter < numIterations; iter++) {
        optimizer.minimize(() => {
          const pred = predict(xs);
          return loss(pred, ys);
        });
      }
    }

    // テストデータがないと話にならないので、テストデータを作る。
    // サンプルにあったテストデータ作成関数だ。
    // numPointsでデータ個数を指定
    // coeffで「a,b,c,d」の定数を与えている。
    // sigmaはランダムノイズを与えるときの定数？なのかな？
    function generateData(numPoints, coeff, sigma = 0.04) {
      return tf.tidy(() => {
        const [a, b, c, d] = [
          tf.scalar(coeff.a), tf.scalar(coeff.b), tf.scalar(coeff.c),
          tf.scalar(coeff.d)
        ];

        const xs = tf.randomUniform([numPoints], -1, 1);

        const ys = a.mul(xs.pow( tf.scalar(3, 'int32')))
          .add(b.mul(xs.square()))
          .add(c.mul(xs))
          .add(d)
          .add(tf.randomNormal([numPoints], 0, sigma)); // ノイズ付加
        return {
          xs,
          ys,
        };
      })
    }

    // a = -3, b = 2, c = -4, d = 7 を定義する。
    const eff = {
      a: -3,
      b: 2,
      c: -4,
      d: 7
    };
    console.log('与えた定数');
    console.log(`a: ${eff.a}`);
    console.log(`b: ${eff.b}`);
    console.log(`c: ${eff.c}`);
    console.log(`d: ${eff.d}`);

    //試験データ作成
    const r1 = generateData(20, eff);
    console.log('試験データ：x');
    r1.xs.print();
    console.log('試験データ：y');
    r1.ys.print();

    // 学習
    console.log('学習開始');
    train(r1.xs, r1.ys, numIterations);

    console.log('学習後');
    // 学習の結果はどうか
    a.print();  // -3に近い値
    b.print();  // 2に近い値
    c.print();  // -4に近い値
    d.print();  // 7に近い値

