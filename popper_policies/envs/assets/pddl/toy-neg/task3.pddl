(define (problem toy-delivery) (:domain toy-delivery)
  (:objects
    loc0 loc1 loc2 loc3
  )
  (:init 
	(at-robby loc1)
  (satisfied loc0)
  )
  (:goal (and (satisfied loc0) (satisfied loc2)))
)
