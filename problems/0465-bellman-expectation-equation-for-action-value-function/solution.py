def bellman_q_value(states, actions, transition_probs, rewards, policy, gamma, theta=1e-6):
    """
    Computes the action-value function Q^pi(s, a) for all state-action pairs
    using iterative policy evaluation until convergence.
    """
    # Initialize Q-values for all state-action pairs to 0.0
    Q = {(s, a): 0.0 for s in states for a in actions}
    
    while True:
        delta = 0.0
        new_Q = Q.copy()
        
        for s in states:
            for a in actions:
                q_new = 0.0
                
                # Sum over possible next states s'
                for s_prime in states:
                    p = transition_probs.get((s, a, s_prime), 0.0)
                    if p == 0.0:
                        continue
                        
                    r = rewards.get((s, a, s_prime), 0.0)
                    
                    # Compute expected value V^pi(s') under policy pi
                    v_s_prime = 0.0
                    for a_prime in actions:
                        pi_val = policy.get((s_prime, a_prime), 0.0)
                        v_s_prime += pi_val * Q[(s_prime, a_prime)]
                        
                    q_new += p * (r + gamma * v_s_prime)
                
                delta = max(delta, abs(q_new - Q[(s, a)]))
                new_Q[(s, a)] = q_new
                
        Q = new_Q
        if delta < theta:
            break
            
    # Round results to 4 decimal places
    return {k: round(v, 4) for k, v in Q.items()}