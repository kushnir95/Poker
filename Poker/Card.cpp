#include "Card.h"

Card::Card()
{
	suit = SPADE;
	rank = TWO;
}

Card::Card(Suits cardSuit, Ranks cardRank)
{
	suit = cardSuit;
	rank = cardRank;
}

int Card::getSuit() const
{
	return suit;
}

int Card::getRank() const
{
	return rank;
}

void Card::setRank(Ranks newRank)
{
	rank = newRank;
}

void Card::setSuit(Suits newSuit)
{
	suit = newSuit;
}

bool Card::operator<(const Card& compCard)
{
	return rank < compCard.getRank() || ((rank == compCard.getRank()) && (suit < compCard.getSuit()));
}

Card::Combination Card::getCombination(std::vector<Card> cardSet)
{
	if (cardSet.size() != 5)
	{
		return Combination::HIGH_CARD;
	}

	sort(cardSet.begin(), cardSet.end());

	//check whether combination is ROYAL FLUSH 
	if ((cardSet[0].getRank() == Ranks::TEN) && (cardSet[1].getRank() == Ranks::JACK) && (cardSet[2].getRank() == Ranks::QUEEN) && (cardSet[3].getRank() == Ranks::KING)
		&& (cardSet[4].getRank() == Ranks::ACE) && (cardSet[0].getSuit() == cardSet[1].getSuit()) && (cardSet[1].getSuit() == cardSet[2].getSuit())
		&& (cardSet[2].getSuit() == cardSet[3].getSuit()) && (cardSet[3].getSuit() == cardSet[4].getSuit()))
	{
		return Combination::ROYAL_FLUSH;
	}

	//check whether combination is STRAIGHT_FLUSH

	if ((((cardSet[0].getRank() == Ranks::TWO) && (cardSet[1].getRank() == Ranks::THREE) && (cardSet[2].getRank() == Ranks::FOUR) && (cardSet[3].getRank() == Ranks::FIVE) && (cardSet[4].getRank() == Ranks::ACE))
		|| ((cardSet[0].getRank() < cardSet[1].getRank()) && (cardSet[0].getRank() + 1 == cardSet[1].getRank()) && (cardSet[1].getRank() < cardSet[2].getRank()) && (cardSet[1].getRank() + 1 == cardSet[2].getRank())
			&& (cardSet[2].getRank() < cardSet[3].getRank()) && (cardSet[2].getRank() + 1 == cardSet[3].getRank()) && (cardSet[3].getRank() < cardSet[4].getRank()) && (cardSet[3].getRank() + 1 == cardSet[4].getRank())))
		&& ((cardSet[0].getSuit() == cardSet[1].getSuit()) && (cardSet[1].getSuit() == cardSet[2].getSuit())
			&& (cardSet[2].getSuit() == cardSet[3].getSuit()) && (cardSet[3].getSuit() == cardSet[4].getSuit())))
	{
		return Combination::ROYAL_FLUSH;
	}

	//check whether combination is FOUR_OF_A_KIND

	if (((cardSet[0].getRank() == cardSet[1].getRank()) && (cardSet[1].getRank() == cardSet[2].getRank()) && (cardSet[2].getRank() == cardSet[3].getRank()))
		|| ((cardSet[1].getRank() == cardSet[2].getRank()) && (cardSet[2].getRank() == cardSet[3].getRank()) && (cardSet[3].getRank() == cardSet[4].getRank())))
	{
		return Combination::FOUR_OF_A_KIND;
	}

	//check whether combination is FULL_HOUSE
	if (((cardSet[0].getRank() == cardSet[1].getRank()) && (cardSet[1].getRank() == cardSet[2].getRank()) && (cardSet[3].getRank() == cardSet[4].getRank()))
		|| ((cardSet[0].getRank() == cardSet[1].getRank()) && (cardSet[2].getRank() == cardSet[3].getRank()) && (cardSet[3].getRank() == cardSet[4].getRank())))
	{
		return Combination::FULL_HOUSE;
	}

	//check whether combination is FLUSH
	if ((cardSet[0].getSuit() == cardSet[1].getSuit()) && (cardSet[1].getSuit() == cardSet[2].getSuit()) && (cardSet[2].getSuit() == cardSet[3].getSuit()) && (cardSet[3].getSuit() == cardSet[4].getSuit()))
	{
		return Combination::FLUSH;
	}

	//check whether combination is STRAIGHT
	if ((((cardSet[0].getRank() == Ranks::TWO) && (cardSet[1].getRank() == Ranks::THREE) && (cardSet[2].getRank() == Ranks::FOUR) && (cardSet[3].getRank() == Ranks::FIVE) && (cardSet[4].getRank() == Ranks::ACE))
		|| ((cardSet[0].getRank() < cardSet[1].getRank()) && (cardSet[0].getRank() + 1 == cardSet[1].getRank()) && (cardSet[1].getRank() < cardSet[2].getRank()) && (cardSet[1].getRank() + 1 == cardSet[2].getRank())
			&& (cardSet[2].getRank() < cardSet[3].getRank()) && (cardSet[2].getRank() + 1 == cardSet[3].getRank()) && (cardSet[3].getRank() < cardSet[4].getRank()) && (cardSet[3].getRank() + 1 == cardSet[4].getRank()))))
	{
		return Combination::STRAIGHT;
	}

	//check whether combination is THREE_OF_A_KIND

	if (((cardSet[0].getRank() == cardSet[1].getRank()) && (cardSet[1].getRank() == cardSet[2].getRank()))
		|| ((cardSet[1].getRank() == cardSet[2].getRank()) && (cardSet[2].getRank() == cardSet[3].getRank()))
		|| ((cardSet[2].getRank() == cardSet[3].getRank()) && (cardSet[3].getRank() == cardSet[4].getRank())))
	{
		return Combination::THREE_OF_A_KIND;
	}

	//check whether combination is TWO_PAIRS

	if (((cardSet[0].getRank() == cardSet[1].getRank()) && (cardSet[2].getRank() == cardSet[3].getRank()))
		|| ((cardSet[0].getRank() == cardSet[1].getRank()) && (cardSet[3].getRank() == cardSet[4].getRank()))
		|| ((cardSet[1].getRank() == cardSet[2].getRank()) && (cardSet[3].getRank() == cardSet[4].getRank())))
	{
		return Combination::TWO_PAIRS;
	}

	//check whether combination is TWO_PAIRS
	if ((cardSet[0].getRank() == cardSet[1].getRank()) || (cardSet[1].getRank() == cardSet[2].getRank()) || (cardSet[2].getRank() == cardSet[3].getRank()) || (cardSet[3].getRank() == cardSet[4].getRank()))
	{
		return Combination::PAIR;
	}

	return Combination::HIGH_CARD;
}


Card::~Card()
{
}
