(define-module (barley env)
  #:use-module (barley ffi)
  #:use-module (ice-9 threads)
  #:export (current-bcl-env shutdown-bcl-env!))

; the BclEnv *, not exported
(define %env #f)
(define %lock (make-mutex))

(define (device-from-environment)
  (let ((v (getenv "BARLEY_DEVICE")))
    (cond ((equal? v "cpu") BCL_DEVICE_TYPE_CPU)
          ((equal? v "gpu") BCL_DEVICE_TYPE_GPU)
          (else BCL_DEVICE_TYPE_ANY))))

(define (init-bcl-env)
  (call-with-bcl-error 'env-alloc
    (lambda (err) (%bcl-env-alloc (device-from-environment) err))))

(define (current-bcl-env)
  (with-mutex %lock
    (unless %env
      (set! %env (init-bcl-env)))
    %env))

(define (shutdown-bcl-env!)
  (with-mutex %lock
    (when %env
      (%bcl-env-free %env)
      (set! %env #f))))
