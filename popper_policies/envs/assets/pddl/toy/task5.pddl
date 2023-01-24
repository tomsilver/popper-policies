(define (problem toy-delivery) (:domain toy-delivery)
  (:objects
    loc0 loc1 loc2
  )
  (:init 
	(at-robby loc2)
  )
  (:goal (and (satisfied loc0)))
)
