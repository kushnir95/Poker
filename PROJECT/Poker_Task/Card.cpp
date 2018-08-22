#include "Card.h"

Card::Card()
{
	suit = SPADE;
	rank = TWO;
}

Card::Card(int cardSuit, int cardRank)
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

void Card::setRank(int newRank)
{
	rank = newRank;
}

void Card::setSuit(int newSuit)
{
	suit = newSuit;
}

bool Card::operator<(const Card& compCard)
{
	return rank < compCard.getRank() || ((rank == compCard.getRank()) && (suit < compCard.getSuit()));
}

std::string Card::getCombination(std::vector<Card> cardSet)
{
	if (cardSet.size() != 5)
	{
		return "Undefined combination";
	}

	sort(cardSet.begin(), cardSet.end());

	//check whether combination is ROYAL FLUSH 
	if ((cardSet[0].getRank() == (int)(Ranks::TEN)) && (cardSet[1].getRank() == (int)(Ranks::JACK)) && (cardSet[2].getRank() == (int)(Ranks::QUEEN)) && (cardSet[3].getRank() == (int)(Ranks::KING))
		&& (cardSet[4].getRank() == (int)(Ranks::ACE)) && (cardSet[0].getSuit() == cardSet[1].getSuit()) && (cardSet[1].getSuit() == cardSet[2].getSuit())
		&& (cardSet[2].getSuit() == cardSet[3].getSuit()) && (cardSet[3].getSuit() == cardSet[4].getSuit()))
	{
		return "ROYAL_FLUSH";
	}

	//check whether combination is STRAIGHT_FLUSH

	if ((((cardSet[0].getRank() == (int)(Ranks::TWO)) && (cardSet[1].getRank() == (int)(Ranks::THREE)) && (cardSet[2].getRank() == (int)(Ranks::FOUR)) && (cardSet[3].getRank() == (int)(Ranks::FIVE)) && (cardSet[4].getRank() == (int)(Ranks::ACE)))
		|| ((cardSet[0].getRank() + 1 == cardSet[1].getRank()) && (cardSet[1].getRank() + 1 == cardSet[2].getRank())
			 && (cardSet[2].getRank() + 1 == cardSet[3].getRank()) && (cardSet[3].getRank() + 1 == cardSet[4].getRank())))
		&& ((cardSet[0].getSuit() == cardSet[1].getSuit()) && (cardSet[1].getSuit() == cardSet[2].getSuit())
			&& (cardSet[2].getSuit() == cardSet[3].getSuit()) && (cardSet[3].getSuit() == cardSet[4].getSuit())))
	{
		return "STRAIGHT_FLUSH";
	}

	//check whether combination is FOUR_OF_A_KIND

	if (((cardSet[0].getRank() == cardSet[1].getRank()) && (cardSet[1].getRank() == cardSet[2].getRank()) && (cardSet[2].getRank() == cardSet[3].getRank()))
		|| ((cardSet[1].getRank() == cardSet[2].getRank()) && (cardSet[2].getRank() == cardSet[3].getRank()) && (cardSet[3].getRank() == cardSet[4].getRank())))
	{
		return "FOUR_OF_A_KIND";
	}

	//check whether combination is FULL_HOUSE
	if (((cardSet[0].getRank() == cardSet[1].getRank()) && (cardSet[1].getRank() == cardSet[2].getRank()) && (cardSet[3].getRank() == cardSet[4].getRank()))
		|| ((cardSet[0].getRank() == cardSet[1].getRank()) && (cardSet[2].getRank() == cardSet[3].getRank()) && (cardSet[3].getRank() == cardSet[4].getRank())))
	{
		return "FULL_HOUSE";
	}

	//check whether combination is FLUSH
	if ((cardSet[0].getSuit() == cardSet[1].getSuit()) && (cardSet[1].getSuit() == cardSet[2].getSuit()) && (cardSet[2].getSuit() == cardSet[3].getSuit()) && (cardSet[3].getSuit() == cardSet[4].getSuit()))
	{
		return "FLUSH";
	}

	//check whether combination is STRAIGHT
	if ((((cardSet[0].getRank() == (int)(Ranks::TWO)) && (cardSet[1].getRank() == (int)(Ranks::THREE)) && (cardSet[2].getRank() == (int)(Ranks::FOUR)) && (cardSet[3].getRank() == (int)(Ranks::FIVE)) && (cardSet[4].getRank() == (int)(Ranks::ACE)))
		|| ((cardSet[0].getRank() < cardSet[1].getRank()) && (cardSet[0].getRank() + 1 == cardSet[1].getRank()) && (cardSet[1].getRank() < cardSet[2].getRank()) && (cardSet[1].getRank() + 1 == cardSet[2].getRank())
			&& (cardSet[2].getRank() < cardSet[3].getRank()) && (cardSet[2].getRank() + 1 == cardSet[3].getRank()) && (cardSet[3].getRank() < cardSet[4].getRank()) && (cardSet[3].getRank() + 1 == cardSet[4].getRank()))))
	{
		return "STRAIGHT";
	}

	//check whether combination is THREE_OF_A_KIND

	if (((cardSet[0].getRank() == cardSet[1].getRank()) && (cardSet[1].getRank() == cardSet[2].getRank()))
		|| ((cardSet[1].getRank() == cardSet[2].getRank()) && (cardSet[2].getRank() == cardSet[3].getRank()))
		|| ((cardSet[2].getRank() == cardSet[3].getRank()) && (cardSet[3].getRank() == cardSet[4].getRank())))
	{
		return "THREE_OF_A_KIND";
	}

	//check whether combination is TWO_PAIRS

	if (((cardSet[0].getRank() == cardSet[1].getRank()) && (cardSet[2].getRank() == cardSet[3].getRank()))
		|| ((cardSet[0].getRank() == cardSet[1].getRank()) && (cardSet[3].getRank() == cardSet[4].getRank()))
		|| ((cardSet[1].getRank() == cardSet[2].getRank()) && (cardSet[3].getRank() == cardSet[4].getRank())))
	{
		return "TWO_PAIRS";
	}

	//check whether combination is TWO_PAIRS
	if ((cardSet[0].getRank() == cardSet[1].getRank()) || (cardSet[1].getRank() == cardSet[2].getRank()) || (cardSet[2].getRank() == cardSet[3].getRank()) || (cardSet[3].getRank() == cardSet[4].getRank()))
	{
		return "PAIR";
	}

	return "HIGH_CARD";
}

std::string Card::toString()
{
	std::string result;
	switch (rank) {
	case(TWO): {
		result = "TWO";
		break;
	}
	case(THREE): {
		result = "THREE";
		break;
	}
	case(FOUR): {
		result = "FOUR";
		break;
	}
	case(FIVE): {
		result = "FIVE";
		break;
	}
	case(SIX): {
		result = "SIX";
		break;
	}
	case(SEVEN): {
		result = "SEVEN";
		break;
	}
	case(EIGHT): {
		result = "EIGHT";
		break;
	}
	case(NINE): {
		result = "NINE";
		break;
	}
	case(TEN): {
		result = "TEN";
		break;
	}
	case(JACK): {
		result = "JACK";
		break;
	}
	case(QUEEN): {
		result = "QUEEN";
		break;
	}
	case(KING): {
		result = "KING";
		break;
	}
	case(ACE): {
		result = "ACE";
		break;
	}
	}

	switch (suit) {
	case(SPADE): {
		result += "_OF_SPADES";
		break;
	}
	case(CLUB): {
		result += "_OF_CLUBS";
		break;
	}
	case(HEART): {
		result += "_OF_HEARTS";
		break;
	}
	case(DIAMOND): {
		result += "_OF_DIAMONDS";
	}
	}

	return result;
}


Card::~Card()
{
}
