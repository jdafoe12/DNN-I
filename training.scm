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

(define images-port (open-input-file "/home/jdafoe/DNN-1/train-images-idx3-ubyte" #:binary #t))
(define labels-port (open-input-file "/home/jdafoe/DNN-1/train-labels-idx1-ubyte" #:binary #t))

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

;TODO: make sure biases work??

; The following matrix multiplication code is from my solution to SICP Exercise 2.37
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

(define (matrix-*-vector m v) ; Dot product called here
  (map (lambda (x) (dot-product x v)) m))

(define (transpose mat)
  (accumulate-n cons '() mat))

(define (matrix-*-matrix m n) ; Calls matrix-*-vector, which calls Dot product. I think this is where my problem is!
  (let ((cols (transpose n)))
    (map (lambda (x) 
           (matrix-*-vector cols x)) 
         m)))


(define (tree-map proc . trees)
  (apply map
         (lambda args
           (if (pair? (car args))  ;; if the first element is a list (sub-tree)
               (apply tree-map proc args)  ;; recursively call tree-map on the sub-lists
               (apply proc args)))  ;; apply the procedure to the corresponding elements
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
  (define (make-zero-list n) (make-list n 0))  ; Helper function to create a list of zeros
  (let ((encoded (make-zero-list 10)))  ; Create a list of 10 zeros
    (list-set! encoded digit 1)  ; Set the value at the index corresponding to the digit to 1
    encoded))  ; Return the one-hot encoded list


(define (load-next-image) ; We need to include a 1 at the end for biases?
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
	(append (remove-last input) (append (list (list (car (car (reverse input))) (softmax (car (car (reverse input)))))))) ; is the remove necessary? i think not. 1 is not added to zl.
	(training-forward-pass (cdr layers) 
				  (training-layer-forward input (car layers) activation-function)
				  activation-function)))

(define (forward-pass layers input activation-function)
  (cadr (car (reverse (training-forward-pass layers input activation-function)))))


; END FORWARD-PASS

; Below are the training specific procedures

(define (mean-squared-error predicted-pd actual-pd n) 
  (/ n (apply + (map (lambda (predicted actual)
					   (expt (- predicted actual) 2)) 
					 predicted-pd 
					 actual-pd))))

(define (cross-entropy-loss predicted-pd actual-pd n) ; For multiple samples, you sum the cross entropy loss???
  (- 0 (apply + (map (lambda (predicted actual)
					   (* actual (log predicted)))
					 predicted-pd
					 actual-pd))))

(define (compute-gradients layers input expected-output activation-function)
  (let* ((model-output (training-forward-pass layers input activation-function))
         (output-layer-output (car (reverse model-output)))  ; Last layer's output
         (output-error (compute-output-error output-layer-output expected-output activation-function))  ; Compute δ^L
         (output-layer-gradients (compute-gradients-for-layer (cadr (last (remove-last model-output))) output-error))  ; Compute last layer's gradients
         (hidden-layer-gradients (backpropagate-layers (reverse layers) (reverse (remove-last model-output)) output-error activation-function))) ; Compute rest
    (reverse (append (list output-layer-gradients) hidden-layer-gradients))))  ; Ensure last layer gradients are included

(define (softmax-derivative z)
  (let* ((aL (softmax z))
         (n (length z)))
    (define (derivative i k)
      (if (= i k)
          (* (list-ref aL i) (- 1 (list-ref aL i)))
          (* (- (list-ref aL i)) (list-ref aL k))))
    (define (generate-jacobian i)
      (map (lambda (k) (derivative i k)) (iota n)))
    (map generate-jacobian (iota n))))

(define (compute-output-error layer-output expected-output activation-function)
  (let* ((zL (car layer-output))
        (aL (cadr layer-output))
        (jacobian (softmax-derivative zL)) 
        (error (map - aL expected-output)))
        (matrix-*-vector (transpose jacobian) error)))

(define (backpropagate-layers layers model-output output-error activation-function)
  (define (backpropagate layers model-output err)
    (if (null? (cdr model-output))
        '()
        (let* ((next-layer-output (car model-output))
               (prev-layer-output (if (null? (cdr model-output)) '() (cadr model-output)))
               (next-layer (car layers))
               (weights (car next-layer))
               (biases (cadr next-layer))
               (zi (car next-layer-output))
               (ai (cadr prev-layer-output))
               (delta (compute-delta weights err zi activation-function))
               (gradients (compute-gradients-for-layer ai delta)))
          (cons gradients (backpropagate (cdr layers) (cdr model-output) delta)))))
  (backpropagate layers model-output output-error))

(define (compute-delta weights err zi activation-function)
  (let ((weighted-error (matrix-*-vector (transpose weights) err)))
    (map (lambda (e z)
           (* e (ReLU-derivative z)))
         weighted-error zi)))

(define (compute-gradients-for-layer ai delta)
  (let ((weight-gradients (outer-product delta ai))
        (bias-gradients delta))
       (list weight-gradients bias-gradients)))

(define (update-weights layers gradients learning-rate)
  (map (lambda (layer gradient) 
		 (list (matrix-minus-matrix (car layer) (matrix-*-scalar (car gradient) learning-rate))
			   (matrix-minus-matrix (cadr layer) (matrix-*-scalar (cadr gradient) learning-rate)))) 
	   layers gradients))

(define (train-step layers input expected-output learning-rate)
  (update-weights layers (compute-gradients layers input expected-output ReLU) learning-rate))

(define (train-epoch layers num-steps learning-rate)
  (define (train-step-iter curr-layers remaining-steps)
    (if (= remaining-steps 0)
        curr-layers
		(let* ((input (list (list 0 (load-next-image)))) 
               (expected-output (load-next-label))
			   (new-weights (train-step curr-layers input expected-output learning-rate)))
		  (train-step-iter new-weights (- remaining-steps 1)))))   
  (train-step-iter layers num-steps))


(define (train-mnist layers epochs num-steps learning-rate)
  (define (train-loop epoch curr-layers)
    (if (= epoch epochs)
        curr-layers 
        (begin
          (display "Training epoch: ") (display epoch) (newline)
          (let ((new-layers (train-epoch curr-layers num-steps learning-rate)))
            (train-loop (+ epoch 1) new-layers)))))
  (train-loop 0 layers))


(define (ReLU-derivative z)
  (if (> z 0) 1 0))

(define layers (list (initialize-layer 784 32)
					 (initialize-layer 32 32)
					 (initialize-layer 32 32)
					 (initialize-layer 32 10)))

(define (save-model layers filename)
  (with-output-to-file filename
    (lambda ()
      (pretty-print layers))))

(define (load-model filename)
  (with-input-from-file filename
    (lambda ()
      (read))))

(save-model (train-mnist layers 6 10000 0.01) "trained-model")


