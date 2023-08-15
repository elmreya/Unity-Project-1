using System.Collections.Generic;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;
using System.Collections;
using UnityEngine;
using Unity.MLAgents;

// General Strategy

// A complicated task like this requires a robust machine learning training method.
// It also requires sufficient time for training.
// As we explored different learning methods and strategies to our ML method, we also made and frequently updated a hard-coded physical symbol system algorithm.
// The behaviour of our ML trained robot guided our strategy for the movement and behaviour for our robot, which is the robot that we will be using for our final project.
// In short, with our strategy, our robot will collect balls efficiently, and once it has collected (collected and deposited at the home base) more balls than the opponent, it will focus on shooting lasers at the opponent.
// This strategy has always been able to beat our ML agents.

// Implementation

// While playing the game, the agent is faced with two different tasks: locomotion and strategy.
// These problems are suited to different training styles and observations.
// Both tasks are difficult to solve on their own, but in combination, the entire problem seems almost impossible.
// We think this is one reason why our previous strategies failed.
// In order to simplify the problem, we removed the complex locomotive actions like turning and movement.
// Instead, we use only the shortcut actions (GoToNearestTarget(), ReturnToBase(), AttackEnemy()) that encapsulate these tasks in higher-level representations.
// This effectively eliminates the locomotion task. The agent no longer needs to figure out how to move through the environment based on velocity and rotation.
// The agent's primary task is deciding when to enter each state.
// According to Clark (2013), connectionist systems are "good at frisbee, bad at logic." This problem requires a logical strategy.
// Therefore, we decided to give Blendo limited movement in order to give him a solid strategy through a hard-coded physical symbol system.
// We hope that this trade-off will be more successful than an agent that is good at movement, but bad at logic.

// Blendo follows simple rules:
// If the enemy agent is within range of the laser, then attack.
// If Blendo is carrying less then one ball, collect the nearest target.
// If Blendo has most of the balls (>= 5) then attack.

public class Blendo : CogsAgent {

    // ------------------------ Global Variables ------------------------ //

    private enum Difficulty {Disabled, Manual, Expert};

    [Header("Heuristic Settings")]
    [SerializeField]
    private Difficulty selection = Difficulty.Expert;
    [SerializeField]
    private float stoppingDistance = 9.2;
    [SerializeField]
    private float stoppingAngle = 5.6;

    // ------------------------ Unity Engine Functions ------------------------ //
    
    protected override void Start() {

        base.Start();
        rewardDict = new Dictionary<string, float>();

        // Punish Blendo for getting hit by a laser.
        rewardDict.Add("frozen", Academy.Instance.EnvironmentParameters.GetWithDefault("frozen", 0.0f));

    }

    protected override void FixedUpdate() {

        base.FixedUpdate();
        LaserControl();
        moveAgent(dirToGo, rotateDir);

    }

    protected override void OnTriggerEnter(Collider collision) {

        base.OnTriggerEnter(collision);

    }

    protected override void OnCollisionEnter(Collision collision) {

        base.OnCollisionEnter(collision);
        
    }

    // ------------------------ MLAgents Functions ------------------------ //

    public override void CollectObservations(VectorSensor sensor) {

        // How far is the enemy?
        sensor.AddObservation(Vector3.Distance(transform.localPosition, enemy.transform.localPosition));

        // How far is my base?
        sensor.AddObservation(Vector3.Distance(transform.localPosition, myBase.transform.localPosition));

        // How far is the enemy to my base?
        sensor.AddObservation(Vector2.Distance(enemy.transform.localPosition, myBase.transform.localPosition));

        // How far is the nearest target?
        GameObject nearestTarget = GetNearestTarget();
        sensor.AddObservation((nearestTarget == null) ? -1 : Vector3.Distance(transform.localPosition, GetNearestTarget().transform.localPosition));

        // What is my current speed?
        sensor.AddObservation(transform.InverseTransformDirection(rBody.velocity).sqrMagnitude);

        // What is the angle formed by my forward heading and the direction from my position to the enemy agent?
        Vector3 directionToEnemy = enemy.transform.localPosition - transform.localPosition;
        sensor.AddObservation(Vector3.Angle(transform.forward, directionToEnemy) / 180);

        // What is the angle formed by my enemy's forward heading and the direction to my position from the enemy agent?
        Vector3 directionToAgent = transform.localPosition - enemy.transform.localPosition;
        sensor.AddObservation(Vector3.Angle(enemy.transform.forward, directionToAgent) / 180);

        // How much time is left?
        sensor.AddObservation(timer.GetComponent<Timer>().GetTimeRemaning());

        // Is the enemy frozen?
        sensor.AddObservation(enemy.GetComponent<CogsAgent>().IsFrozen());

        // How many targets are in my base?
        sensor.AddObservation(myBase.GetComponent<HomeBase>().GetCaptured());

        // How many targets am I carrying?
        sensor.AddObservation(GetCarrying());

        // How many targets is the enemy carrying?
        sensor.AddObservation(enemy.GetComponent<CogsAgent>().GetCarrying());

    }

    public override void OnActionReceived(ActionBuffers actions) {

        var act = actions.DiscreteActions;

        MovePlayer((int)act[0]);

        // Contantly add a small negative reward over time.
        AddReward(Academy.Instance.EnvironmentParameters.GetWithDefault("life_drain", 0.0f));
    }

    public override void Heuristic(in ActionBuffers actionsOut) {

        var discreteActionsOut = actionsOut.DiscreteActions;

        if (selection == Difficulty.Disabled) {

        } else if (selection == Difficulty.Manual) {
            ManualControl(discreteActionsOut);

        } else if (selection == Difficulty.Expert) {
            ExpertMode(discreteActionsOut);

        }

    }

    // ------------------------ Heuristic Helper Functions ------------------------ //
  
    void ManualControl(ActionSegment<int> actions) {

        if (Input.GetKey(KeyCode.Space)){
            actions[0] = 0;
        }
        if (Input.GetKey(KeyCode.Z)){
            actions[0] = 1;
        }       
        if (Input.GetKey(KeyCode.X)){
            actions[0] = 2;

        }

    }

    void ExpertMode(ActionSegment<int> actions) {

        int collectedCount = myBase.GetComponent<HomeBase>().GetCaptured();
        int carriedCount = GetCarrying();
        float attackDistance = 0;
        float distanceToEnemy = Vector3.Distance(enemy.transform.localPosition, transform.localPosition);

        if (carriedCount == 0) {
            attackDistance = 25;
            
        } else {
            attackDistance = 10;

        }

        if (carriedCount < 1) {
            actions[0] = 0;
            
        } else {
            actions[0] = 2;

        }
        
        if (distanceToEnemy < attackDistance && !enemy.GetComponent<CogsAgent>().IsFrozen()) {

            actions[0] = 1;

        }
        
        if (collectedCount >= 5) {
            actions[0] = 1;

        }

    }

    // ------------------------ Movement Helper Functions ------------------------ //

    private void MovePlayer(int decision) {
    
        dirToGo = Vector3.zero;
        rotateDir = Vector3.zero;

        if (decision == 0) {
            GoToNearestTarget();
        }

        if (decision == 1) {
            RotateAndShoot(enemy.transform.localPosition);
        }

        if (decision == 2) {
            GoToBase();
        }
        
    }

    void GoToNearestTarget() {
        GameObject nearestTarget = GetNearestTarget();

        if (nearestTarget != null) {
            TurnAndGo(nearestTarget.transform.localPosition);

        }

    }

    void GoToBase() {
        TurnAndGo(myBase.transform.localPosition);

    }

    private void RotateAndShoot(Vector3 location) {
        TurnAndGoForward(location);

        RaycastHit hit;
        if (Physics.Raycast(transform.position, transform.forward, out hit, 20)) {
            GameObject hitObj = hit.collider.gameObject;
            
            if (hitObj.CompareTag("Player") && hitObj != this.gameObject) {
                
                if (!hitObj.GetComponent<CogsAgent>().IsFrozen()) {
                    StartCoroutine("PulseLaser");

                }

            }

        }

    }

    private IEnumerator PulseLaser() 
    {
        SetLaser(true);
        yield return new WaitForSeconds(0.1f);
        SetLaser(false);

    }

    /// <summary>
    /// Blendo will move and rotate so that his forward transform will face the given location.
    /// </summary>
    private void TurnAndGoForward(Vector3 location) {

        Vector3 distanceToLocation = location - transform.localPosition;
        Vector3 dirToLocation = (distanceToLocation).normalized;

        float dot = Vector3.Dot(transform.forward, dirToLocation);

        if (dot >= 0) {
            dirToGo = transform.forward;

        } else if (dot < 0) {
            dirToGo = -transform.forward;
            
        }
        
        Breaking(distanceToLocation, transform.forward);
        Rotate(dirToLocation, transform.forward);

    }

    /// <summary>
    /// Blendo will move and rotate so that he will move forwards or backwards to the given position. 
    /// </summary>
    private void TurnAndGo(Vector3 position) {

        Vector3 distanceToPosition = position - transform.localPosition;
        Vector3 dirToPosition = (distanceToPosition).normalized;

        float dot = Vector3.Dot(transform.forward, dirToPosition);
        Vector3 heading = Vector3.zero;

        if (dot >= 0) {
            dirToGo = transform.forward;

        } else {
            dirToGo = -transform.forward;
            
        }
        
        Rotate(dirToPosition, dirToGo);
        Breaking(distanceToPosition, dirToGo);

    }

    /// <summary>
    /// Blendo will slow down if:
    /// He is within a certain distance from the target (stoppingDistance).
    /// And the angle between his desired and current forward heading is greater than (stoppingAngle)
    /// </summary>
    private void Breaking(Vector3 dirToLocation, Vector3 heading) {

        float angle = Vector3.Angle(dirToGo, dirToLocation);
        float distScalar = dirToLocation.magnitude;

        if (distScalar < stoppingDistance && angle > stoppingAngle) {
            dirToGo = Vector3.zero;

        }

    }

    /// <summary>
    /// Rotate towards a given direction with respect to a desired forward heading (backwards or forwards).
    /// </summary>
    private void Rotate(Vector3 dirToTarget, Vector3 heading) {

        float rotation = Vector3.SignedAngle(dirToTarget, heading, Vector3.up);
        if (rotation < -0.5) {
            rotateDir = transform.up;;

        } else if (rotation > 0.5) {
            rotateDir = -transform.up;

        }

    }

    protected GameObject GetNearestTarget() {

        GameObject nearestTarget = null;
        float distance = 200;

        foreach (var target in targets) {

            float currentDistance = Vector3.Distance(target.transform.localPosition, transform.localPosition);
            if (currentDistance < distance && target.GetComponent<Target>().GetCarried() == 0 && target.GetComponent<Target>().GetInBase() != team) {
                distance = currentDistance;
                nearestTarget = target;

            }

        }

        return nearestTarget;

    }

}