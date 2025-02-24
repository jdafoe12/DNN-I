#!/usr/env/bin guile
!#

(use-modules (ice-9 binary-ports))
(use-modules (ice-9 textual-ports))
(use-modules (ice-9 pretty-print))
(use-modules (scheme base))


(define (read-u32 port)
  (let ((b1 (read-u8 port))
        (b2 (read-u8 port))
        (b3 (read-u8 port))
        (b4 (read-u8 port)))
    (+ (ash b1 24) (ash b2 16) (ash b3 8) b4)))

(define images-port (open-input-file "/home/jdafoe/DNN-1/t10k-images-idx3-ubyte" #:binary #t))
(define labels-port (open-input-file "/home/jdafoe/DNN-1/t10k-labels-idx1-ubyte" #:binary #t))

(define images-magic (read-u32 images-port))
(define num-images (read-u32 images-port))
(define num-rows (read-u32 images-port))
(define num-cols (read-u32 images-port))

(define labels-magic (read-u32 labels-port))
(define num-labels (read-u32 labels-port))

(display "Image magic: ") (display images-magic) (newline)
(display "Labels magic: ") (display labels-magic) (newline)
(display "Images: ") (display num-images) (newline);
(display "Rows: ") (display num-rows) (newline)
(display "Cols: ") (display num-cols) (newline)


(define (accumulate op initial sequence)
  (if (null? sequence)
	initial
	(op (car sequence)
		(accumulate op initial (cdr sequence)))))

(define (accumulate-n op init seqs)
  (if (null? (car seqs))
  '()
  (cons (accumulate op init (map car seqs))
        (accumulate-n op init (map cdr seqs)))))


(define (dot-product v w)
  (let ((products (map * v w)))
    (apply + products)))

(define (matrix-*-vector m v) 
  (map (lambda (x) (dot-product x v)) m))

(define (transpose mat)
  (accumulate-n cons '() mat))

(define (matrix-*-matrix m n)
  (let ((cols (transpose n)))
    (map (lambda (x) 
           (matrix-*-vector cols x)) 
         m)))


(define (tree-map proc . trees)
  (apply map
         (lambda args
           (if (pair? (car args)) 
			 (apply tree-map proc args) 
			 (apply proc args))) 
		 trees))

(define (matrix-*-scalar m s)
  (tree-map (lambda (l) (* s l)) m))

(define (matrix-minus-matrix m1 m2)
  (tree-map (lambda (x y) (- x y)) m1 m2))

(define (outer-product v1 v2)
  (map (lambda (x) (map (lambda (y) (* x y)) v2)) v1))

(define (ReLU x)
  (if (< x 0) 0 x))

(define (read-image port)
  (get-bytevector-n port (* num-rows num-cols))) ; 28x28 = 784 bytes

(define (read-label port)
  (read-u8 port))

(define (normalize-image image-vector)
  (map (lambda (pixel) (/ pixel 255.0)) (u8vector->list image-vector)))


(define (one-hot-encode digit)
  (define (make-zero-list n) (make-list n 0))
  (let ((encoded (make-zero-list 10)))
	(list-set! encoded digit 1)
	encoded)) 

(define (load-next-image)
  (normalize-image (read-image images-port)))

(define (load-next-label)
  (one-hot-encode (read-label labels-port)))

(define (initialize-layer input-size output-size)
  (list
   (map (lambda (_) 
          (map (lambda (_) 
                 (+ 0 (* (sqrt (/ 2 input-size)) (random:normal)))) ; He Initialization
               (iota input-size)))
        (iota output-size))   
   (map (lambda (_) 
          (+ 0 (* (sqrt (/ 2 input-size)) (random:normal))))
		(iota output-size))))

(define (softmax logits)
  (let* ((exps (map exp logits))
         (sum (apply + exps)))
    (map (lambda (x) (/ x sum)) exps)))

(define (remove-last l)
  (reverse (cdr (reverse l))))

(define (last l)
  (car (reverse l)))

; FORWARD PASS
(define (training-layer-forward input layer activation-function)
(let* ((weights (car layer))
       (biases (cadr layer))
       (activations (cadr (last input)))
       (zl (map + (matrix-*-vector weights activations) biases))
       (new-activations (map activation-function zl)))
  (append input (list (list zl new-activations)))))


; For the training forward-pass, we need to keep z^l for each layer.

(define (identity x) x)

(define (training-forward-pass layers input activation-function)
  (if (null? layers)
	(append (remove-last input) (append (list (list (car (car (reverse input))) (softmax (car (car (reverse input))))))))
	(training-forward-pass (cdr layers) 
				  (training-layer-forward input (car layers) activation-function)
				  activation-function)))

(define (forward-pass layers input activation-function)
  (cadr (car (reverse (training-forward-pass layers input activation-function)))))

(define (argmax lst)
  (define (argmax-helper lst max-idx max-val curr-idx)
    (if (null? lst)
        max-idx
        (if (> (car lst) max-val)
            (argmax-helper (cdr lst) curr-idx (car lst) (+ curr-idx 1))
            (argmax-helper (cdr lst) max-idx max-val (+ curr-idx 1)))))
  (argmax-helper lst 0 (car lst) 0))

(define (test-model layers num-test-samples)
  (let loop ((remaining-samples num-test-samples)
             (correct 0))
    (if (= remaining-samples 0)
        (begin
          (display "Accuracy: ")
          (display (* 100.0 (/ correct num-test-samples)))
          (display "%\n")
          (/ correct num-test-samples))
        (let* ((input (list (list 0 (load-next-image))))
               (true-label (read-label labels-port))
               (model-output (forward-pass layers input ReLU))
               (predicted-label (argmax model-output)))
          (if (= predicted-label true-label)
              (loop (- remaining-samples 1) (+ correct 1))
              (loop (- remaining-samples 1) correct))))))

(define (load-model filename)
  (with-input-from-file filename
    (lambda ()
      (read))))
(define trained-layers (load-model "/home/jdafoe/DNN-1/trained-model"))

(test-model trained-layers 10000)
