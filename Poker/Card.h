#pragma once
#include <cassert>
#include <vector>
#include <algorithm>

class Card
{

private:
	enum Suits{SPADE = 1, CLUB = 2, HEART = 3, DIAMOND = 4};
	enum Ranks{TWO = 2, THREE = 3, FOUR = 4, FIVE = 5, SIX = 6, SEVEN = 7, EIGHT = 8, NINE = 9, TEN = 10, JACK = 11, QUEEN = 12, KING = 13, ACE = 14};

	Suits suit;
	Ranks rank;

public:

	static enum Combination{HIGH_CARD = 0, PAIR = 1, TWO_PAIRS = 2, THREE_OF_A_KIND = 3, STRAIGHT = 4, FLUSH = 5, FULL_HOUSE = 6, FOUR_OF_A_KIND = 7, STRAIGHT_FLUSH = 8, ROYAL_FLUSH = 9};

	Card();
	Card(Suits suit, Ranks rank);
	
	int getSuit() const;
	int getRank() const;

	void setSuit(Suits newSuit);
	void setRank(Ranks newRank);

	bool operator<(const Card& compCard);
	static Combination getCombination(std::vector<Card> cardSet);

	~Card();
};

