import os
import sys
import anthropic

if not os.environ.get("ANTHROPIC_API_KEY"):
    print("ERROR: ANTHROPIC_API_KEY environment variable is not set.")
    print("Please set it in your terminal before running python agent.py")
    sys.exit(1)

client = anthropic.Anthropic()  
MODEL = "claude-sonnet-5-5"     


def extract_text(response, agent_name: str) -> str:
    """Safely extracts only TextBlocks (Slide 8), ignoring any ThinkingBlocks."""
    text = "".join(b.text for b in response.content if b.type == "text").strip()
    prefix = f"{agent_name}:"
    if text.startswith(prefix):
        text = text[len(prefix):].strip()
    return text


# 1. base code (Slide 11)
class Agent:
    def __init__(self, name: str, role: str):
        self.name = name
        self.system = (
            f"You are {name}. {role} "
            "Reply with your next line only, 1-2 sentences."
        )

    def act(self, board: list) -> str:
        view = "\n".join(f"{who}: {text}" for who, text in board)
        r = client.messages.create(
            model=MODEL,
            max_tokens=1000,  
            system=self.system,
            messages=[{"role": "user", "content": view or "Begin."}],
        )
        return extract_text(r, self.name)


# ex2 (Slides 4 & 14)
class MemoryBuyerAgent(Agent):
    def __init__(self, name: str, role: str):
        super().__init__(name, role)
        self.base_role = role
        self.memories = []

    def remember(self, episode: int, outcome: str, last_lines: list):
        snippet = " | ".join(f"{w}: {t}" for w, t in last_lines)
        self.memories.append(f"Iteration {episode}: {outcome} (Closing exchange: {snippet})")

    def act(self, board: list) -> str:
        memory_prompt = ""
        if self.memories:
            history = "\n".join(f"- {m}" for m in self.memories)
            memory_prompt = (
                f"\n\nMemory of your past negotiations:\n{history}\n"
                "Use this experience to negotiate a lower price than before, "
                "while still ensuring a deal is reached before the turns run out."
            )

        system_with_memory = (
            f"You are {self.name}. {self.base_role}{memory_prompt} "
            "Reply with your next line only, 1-2 sentences."
        )
        view = "\n".join(f"{who}: {text}" for who, text in board)
        r = client.messages.create(
            model=MODEL,
            max_tokens=1000,
            system=system_with_memory,
            messages=[{"role": "user", "content": view or "Begin."}],
        )
        return extract_text(r, self.name)


# 3. round-robin orchestrator (Slide 12)
def run_orchestrator(agents: list, max_turns: int, log_lines: list):
    board = []
    outcome = f"NO CONVERGENCE after {max_turns} turns"

    for turn in range(max_turns):
        speaker = agents[turn % len(agents)]
        line = speaker.act(board)
        board.append((speaker.name, line))

        formatted = f"[Turn {turn + 1}] {speaker.name}: {line}"
        print(formatted)
        log_lines.append(formatted)

        if "DEAL:" in line:
            outcome = f"{line} (settled on turn {turn + 1})"
            break

    summary = f"--> Outcome: {outcome}\n" + ("-" * 65)
    print(summary)
    log_lines.append(summary)
    return board, outcome


# main code: base + ex1 + ex2 
def main():
    log_lines = []

    def header(title: str):
        banner = f"\n{'=' * 65}\n{title}\n{'=' * 65}"
        print(banner)
        log_lines.append(banner)

    # part A: base negotiation (Slide 12)
    header("PART 1: BASELINE NEGOTIATION (Buyer, Seller, Mediator)")
    baseline_agents = [
        Agent("Buyer", "You want a used laptop and will pay at most 600 EUR. "
                       "Never reveal your limit. Make concrete price offers quickly."),
        Agent("Seller", "You sell a used laptop. You will not go below 450 EUR. "
                        "Open at 800 EUR, then make meaningful concessions to close a deal within 9 turns."),
        Agent("Mediator", "You are neutral. When buyer and seller agree on a "
                          "price, reply 'DEAL: <price>'. Otherwise nudge them."),
    ]
    run_orchestrator(baseline_agents, max_turns=9, log_lines=log_lines)

    # part B: ex1 (Slide 14) 
    # "1. Add another seller with a higher bottom price and the same product to sell."
    header("PART 2: EXERCISE 1 (Two Competing Sellers: Floor 450 EUR vs 550 EUR)")
    ex1_agents = [
        Agent("Buyer", "You want a used laptop and will pay at most 600 EUR. "
                       "Never reveal your limit. Compare offers from Seller_1 and Seller_2 "
                       "and explicitly accept the best offer before turn 16."),
        Agent("Seller_1", "You sell a used laptop. You will not go below 450 EUR. "
                          "Open at 800 EUR. Compete aggressively with Seller_2 to win the Buyer."),
        Agent("Seller_2", "You sell the exact same used laptop. You will not go below 550 EUR. "
                          "Open at 800 EUR. Compete aggressively with Seller_1 to win the Buyer."),
        Agent("Mediator", "You are neutral. When the buyer and one of the sellers agree on a "
                          "price, reply 'DEAL: <price> with <Seller_Name>'. Otherwise nudge them."),
    ]
    run_orchestrator(ex1_agents, max_turns=16, log_lines=log_lines)

    # part C: ex2 (Slide 14)
    # "2. Allow the Buyer to have a memory of past negotiations and run several iterations."
    header("PART 3: EXERCISE 2 (Buyer with Persistent Memory over 3 Iterations)")
    memory_buyer = MemoryBuyerAgent(
        "Buyer",
        "You want a used laptop and will pay at most 600 EUR. Never reveal your limit. "
        "Negotiate decisively so you close a deal before turn 9."
    )

    for episode in range(1, 4):
        sub_header = f"\n--- Exercise 2: Iteration {episode} ---"
        print(sub_header)
        log_lines.append(sub_header)

        seller = Agent(
            "Seller",
            "You sell a used laptop. You will not go below 450 EUR. "
            "Open at 800 EUR, but be willing to drop down toward your 450 EUR minimum to close a deal within 9 turns."
        )
        mediator = Agent(
            "Mediator",
            "You are neutral. When buyer and seller agree on a price, "
            "reply 'DEAL: <price>'. Otherwise nudge them."
        )

        board, outcome = run_orchestrator(
            [memory_buyer, seller, mediator],
            max_turns=9,
            log_lines=log_lines,
        )
        memory_buyer.remember(episode, outcome, board[-3:])

    with open("negotiation_output.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))
    print("\n[Saved full transcript to negotiation_output.txt]")


if __name__ == "__main__": 
    main()